
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main


@pytest.fixture(scope="module")
def client():
    """Start the real application and load the trained model once."""
    with TestClient(api_main.app) as test_client:
        yield test_client


def make_image_bytes(image_format="JPEG"):
    """Generate a small in-memory image for API tests."""
    image = Image.new("RGB", (256, 256), color=(80, 150, 60))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Flower Image Classifier API"


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_ready"] is True
    assert response.json()["device"] in ("cpu", "cuda")


def test_classes(client):
    response = client.get("/classes")

    assert response.status_code == 200
    assert response.json() == {
        "classes": [
            "daisy",
            "dandelion",
            "roses",
            "sunflowers",
            "tulips",
        ],
        "count": 5,
    }


def test_predict_valid_image(client):
    response = client.post(
        "/predict",
        files={
            "image": (
                "flower.jpg",
                make_image_bytes("JPEG"),
                "image/jpeg",
            )
        },
        data={"top_k": "3"},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["predicted_label"] in [
        "daisy",
        "dandelion",
        "roses",
        "sunflowers",
        "tulips",
    ]
    assert result["class_index"] in range(5)
    assert len(result["top_k"]) == 3
    assert result["score_type"] == "softmax_score"

    scores = [item["score"] for item in result["top_k"]]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)

    for item in result["top_k"]:
        assert item["label"] in [
            "daisy",
            "dandelion",
            "roses",
            "sunflowers",
            "tulips",
        ]
        assert item["class_index"] in range(5)


def test_predict_empty_upload(client):
    response = client.post(
        "/predict",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_predict_unsupported_image_format(client):
    response = client.post(
        "/predict",
        files={
            "image": (
                "flower.bmp",
                make_image_bytes("BMP"),
                "image/bmp",
            )
        },
    )

    assert response.status_code == 415
    assert "Unsupported image format" in response.json()["detail"]


def test_predict_corrupt_image(client):
    response = client.post(
        "/predict",
        files={
            "image": (
                "corrupt.jpg",
                b"this is not a valid image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
    assert "valid, readable image" in response.json()["detail"]


@pytest.mark.parametrize("top_k", ["0", "6", "-1", "abc"])
def test_predict_invalid_top_k(client, top_k):
    response = client.post(
        "/predict",
        files={
            "image": (
                "flower.jpg",
                make_image_bytes("JPEG"),
                "image/jpeg",
            )
        },
        data={"top_k": top_k},
    )

    assert response.status_code == 422



def test_health_when_model_fails_to_load(monkeypatch):
    original_predictor = getattr(
        api_main.app.state, "predictor", None
    )

    def fail_to_load(*args, **kwargs):
        raise RuntimeError("Simulated model-loading failure")

    monkeypatch.setattr(api_main, "FlowerPredictor", fail_to_load)

    try:
        with TestClient(api_main.app) as test_client:
            response = test_client.get("/health")

        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert response.json()["model_ready"] is False
    finally:
        api_main.app.state.predictor = original_predictor


def test_predict_when_model_is_unavailable(monkeypatch):
    original_predictor = getattr(
        api_main.app.state, "predictor", None
    )

    def fail_to_load(*args, **kwargs):
        raise RuntimeError("Simulated model-loading failure")

    monkeypatch.setattr(api_main, "FlowerPredictor", fail_to_load)

    try:
        with TestClient(api_main.app) as test_client:
            response = test_client.post(
                "/predict",
                files={
                    "image": (
                        "flower.jpg",
                        make_image_bytes("JPEG"),
                        "image/jpeg",
                    )
                },
            )

        assert response.status_code == 503
        assert response.json()["detail"] == (
            "The classification model is not ready."
        )
    finally:
        api_main.app.state.predictor = original_predictor



def test_predict_when_model_is_unavailable(monkeypatch):
    def fail_to_load(*args, **kwargs):
        raise RuntimeError("Simulated model-loading failure")

    monkeypatch.setattr(api_main, "FlowerPredictor", fail_to_load)

    with TestClient(api_main.app) as test_client:
        response = test_client.post(
            "/predict",
            files={
                "image": (
                    "flower.jpg",
                    make_image_bytes("JPEG"),
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The classification model is not ready."
    )

def test_predict_oversized_upload(client):
    # The API limit is 10 MiB. This checks rejection without
    # requiring a real image or an additional download.
    response = client.post(
        "/predict",
        files={
            "image": (
                "oversized.jpg",
                b"x" * (10 * 1024 * 1024 + 1),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


def test_predict_image_exceeding_pixel_limit(client):
    # Create a valid PNG with dimensions above the 20-megapixel limit.
    image = Image.new("RGB", (5000, 5000), color=(20, 40, 60))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/predict",
        files={
            "image": (
                "large.png",
                buffer.getvalue(),
                "image/png",
            )
        },
    )

    assert response.status_code == 413
    assert "20-megapixel" in response.json()["detail"]
