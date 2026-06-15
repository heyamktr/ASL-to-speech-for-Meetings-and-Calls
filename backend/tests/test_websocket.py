from fastapi.testclient import TestClient

from app.main import app

# No-lifespan client: enough for /health and message-validation, which happen
# before the model is touched. (Use `with TestClient(app)` to load the model.)
client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_websocket_echoes_timestamp_and_accepts_144_landmarks() -> None:
    timestamp = 1710000000000
    message = {
        "landmarks": [0.0] * 144,
        "session_id": "test-session",
        "timestamp": timestamp,
    }

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(message)
        response = websocket.receive_json()

    assert response["timestamp"] == timestamp
    assert "error" not in response


def test_websocket_rejects_invalid_landmark_count() -> None:
    message = {
        "landmarks": [0.0] * 63,
        "session_id": "test-session",
        "timestamp": 1710000000000,
    }

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(message)
        response = websocket.receive_json()

    assert response == {"error": "landmarks must be a list of exactly 144 numbers"}


def test_websocket_full_inference_flow() -> None:
    """With the model loaded, a full window of consistent frames yields a
    prediction (timestamp always echoed)."""
    pattern = [0.1] * 144  # non-zero => hand present => active stride
    with TestClient(app) as warm_client:  # `with` runs lifespan -> loads model
        with warm_client.websocket_connect("/ws") as websocket:
            saw_prediction = False
            for i in range(160):
                websocket.send_json(
                    {"landmarks": pattern, "session_id": "flow", "timestamp": i}
                )
                resp = websocket.receive_json()
                assert resp["timestamp"] == i
                assert "error" not in resp
                if "prediction" in resp:
                    saw_prediction = True
                    assert "confidence" in resp

    assert saw_prediction, "expected at least one prediction after a full window"
