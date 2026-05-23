from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_websocket_returns_dummy_prediction_and_echoes_timestamp() -> None:
    timestamp = 1710000000000
    message = {
        "landmarks": [0.0] * 63,
        "session_id": "test-session",
        "timestamp": timestamp,
    }

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(message)
        response = websocket.receive_json()

    assert response == {
        "prediction": "hello",
        "confidence": 0.99,
        "timestamp": timestamp,
    }


def test_websocket_rejects_invalid_landmark_count() -> None:
    message = {
        "landmarks": [0.0] * 62,
        "session_id": "test-session",
        "timestamp": 1710000000000,
    }

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(message)
        response = websocket.receive_json()

    assert response == {"error": "landmarks must be a list of exactly 63 numbers"}
