# WebSocket Contract

Dev B owns the server side of this contract. Dev C owns the browser client side.
Any change to this message format requires a team conversation before code changes.

## Endpoint

Local development:

```text
ws://localhost:8000/ws
```

`/ws` is the canonical Week 1 endpoint. `/ws/predict` currently exists as a
temporary compatibility alias for the original scaffold.

## Incoming Message

Sent from the browser extension to the FastAPI backend.

```json
{
  "landmarks": [0.0, 0.1, 0.2],
  "session_id": "dev-session-123",
  "timestamp": 1710000000000
}
```

Fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `landmarks` | `number[]` | Yes | Exactly 63 numbers: 21 hand landmarks times `x`, `y`, `z`. Raw video frames must never be sent. |
| `session_id` | `string` | Yes | Stable ID for the current extension/user session. |
| `timestamp` | `number` | Yes | Browser-created timestamp. The server echoes this value back so the client can measure round-trip latency. |

## Outgoing Message

Sent from the FastAPI backend to the browser extension.

```json
{
  "prediction": "hello",
  "confidence": 0.99,
  "timestamp": 1710000000000
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `prediction` | `string` | Week 1 returns a dummy prediction. Week 2 replaces this with real model output. |
| `confidence` | `number` | Float from `0.0` to `1.0`. Week 1 dummy server returns `0.99`. |
| `timestamp` | `number` | Exact timestamp from the incoming message. |

## Invalid Messages

If the server receives a malformed message, it keeps the WebSocket open and
returns an error object:

```json
{
  "error": "landmarks must be a list of exactly 63 numbers"
}
```

## Latency Logging

The backend logs one entry for every valid message:

```text
prediction_sent session_id=dev-session-123 frame_values=63 latency_ms=1.23
```

This measures server-side time from JSON receipt through response send.
