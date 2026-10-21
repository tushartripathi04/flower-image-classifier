
import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from httpx import request
from matplotlib import image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api.inference import DEFAULT_CHECKPOINT, FlowerPredictor


# ---------------- Configuration ----------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS = 20_000_000  # 20 megapixels

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

checkpoint_path = Path(
    os.getenv("FLOWER_MODEL_PATH", str(DEFAULT_CHECKPOINT))
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]


# ---------------- Response schemas ----------------

class APIInfo(BaseModel):
    name: str
    version: str
    description: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    device: str | None = None


class ClassesResponse(BaseModel):
    classes: list[str]
    count: int


class TopKPrediction(BaseModel):
    label: str
    class_index: int
    score: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_label: str
    class_index: int
    top_k: list[TopKPrediction]
    score_type: str


# ---------------- Application lifecycle ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model once when the application starts."""
    app.state.predictor = None

    try:
        app.state.predictor = FlowerPredictor(
            checkpoint_path=checkpoint_path
        )
        logger.info(
            "Flower model loaded successfully on %s",
            app.state.predictor.device,
        )
    except Exception:
        # Keep the details in server logs, not in API responses.
        logger.exception("Flower model initialization failed")

    yield

    app.state.predictor = None
    logger.info("Flower API shut down")


app = FastAPI(
    title="Flower Image Classifier API",
    description=(
        "Classify flower images using a trained EfficientNet-B0 model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ---------------- Helpers ----------------

def get_predictor(request: Request) -> FlowerPredictor:
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="The classification model is not ready.",
        )

    return predictor


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    """Read and validate an uploaded image before inference."""

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB upload limit.",
            )

        with Image.open(BytesIO(contents)) as image:
            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format. Use JPEG, PNG, or WEBP.",
                )

            width, height = image.size

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="The image dimensions are invalid.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the 20-megapixel limit.",
                )

            # Force decoding now so truncated/corrupt images are rejected
            # before reaching the model.
            image.load()
            return image.convert("RGB")

    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid, readable image.",
        ) from None
    finally:
        await file.close()


# ---------------- Endpoints ----------------

@app.get("/", response_model=APIInfo)
async def root():
    return APIInfo(
        name="Flower Image Classifier API",
        version="1.0.0",
        description="EfficientNet-B0 flower classification service.",
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "model_ready": False,
                "device": None,
            },
        )

    return HealthResponse(
        status="ok",
        model_ready=True,
        device=str(predictor.device),
    )


@app.get("/classes", response_model=ClassesResponse)
async def classes(request: Request):
    predictor = get_predictor(request)

    return ClassesResponse(
        classes=predictor.class_names,
        count=len(predictor.class_names),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    image: Annotated[UploadFile, File(description="Flower image file")],
    top_k: Annotated[int, Form(ge=1, le=5)] = 3,
):
    pil_image = await read_uploaded_image(image)
    predictor = get_predictor(request)

    # PyTorch inference is blocking; run it in a worker thread rather than
    # blocking FastAPI's asynchronous event loop.
    try:
        result = await run_in_threadpool(
            predictor.predict,
            pil_image,
            top_k,
        )
    except Exception:
        logger.exception("Flower prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try another image.",
        ) from None

    return PredictionResponse(**result)
