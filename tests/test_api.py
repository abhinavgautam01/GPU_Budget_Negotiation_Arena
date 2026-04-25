from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


def test_health_and_tasks() -> None:
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    tasks = client.get("/tasks").json()["tasks"]
    assert {task["task_type"] for task in tasks} == {"single_trade", "market_round", "coalition_market"}


def test_reset_step_state_api() -> None:
    client = TestClient(app)
    reset = client.post("/reset", json={"task_type": "single_trade", "seed": 99})
    assert reset.status_code == 200
    obs = reset.json()["observation"]
    assert obs["task_id"] == "single_trade-99"
    step = client.post("/step", json={"action_type": "wait"})
    assert step.status_code == 200
    assert "observation" in step.json()
    state = client.get("/state")
    assert state.status_code == 200
    assert state.json()["state"]["task_id"] == "single_trade-99"
    assert "labs" not in state.json()["state"]


def test_private_state_requires_debug_flag() -> None:
    client = TestClient(app)
    reset = client.post("/reset", json={"task_type": "single_trade", "seed": 100})
    assert reset.status_code == 200
    state = client.get("/state?include_private=true")
    assert state.status_code == 403
