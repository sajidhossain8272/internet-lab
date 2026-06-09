import json
import time
import pytest
from io import BytesIO

from app import app
from internet_experiment_lab.custom_model import default_custom_spec


def start_response(status, headers):
    start_response.status = status
    start_response.headers = headers


def make_request(path, method="GET", body=None):
    if body is None:
        body = b""
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "CONTENT_TYPE": "application/json",
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "http",
        "wsgi.input": BytesIO(body),
        "wsgi.errors": BytesIO(),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }
    result = app(environ, start_response)
    body_bytes = b"".join(result)
    return start_response.status, body_bytes.decode("utf-8")


def test_api_custom_schema_returns_schema():
    status, body = make_request("/api/custom/schema")
    assert status.startswith("200")
    data = json.loads(body)
    assert "schema" in data
    assert "variables" in data["schema"]


def test_api_custom_validate_endpoint_success():
    spec = default_custom_spec()
    payload = json.dumps({"spec": spec}).encode("utf-8")
    status, body = make_request("/api/custom/validate", method="POST", body=payload)
    assert status.startswith("200")
    data = json.loads(body)
    assert data.get("valid") is True


def test_api_custom_validate_endpoint_failure():
    spec = default_custom_spec()
    spec["variables"][0]["type"] = "bad"
    payload = json.dumps({"spec": spec}).encode("utf-8")
    status, body = make_request("/api/custom/validate", method="POST", body=payload)
    assert status.startswith("400")
    data = json.loads(body)
    assert "error" in data


def test_api_custom_queue_and_status():
    spec = default_custom_spec()
    payload = json.dumps({"spec": spec, "size": 50, "seed": 1}).encode("utf-8")
    status, body = make_request("/api/custom/queue", method="POST", body=payload)
    assert status.startswith("200")
    queue_data = json.loads(body)
    assert "job_id" in queue_data
    job_id = queue_data["job_id"]

    for _ in range(20):
        time.sleep(0.1)
        status, body = make_request(f"/api/custom/status/{job_id}")
        assert status.startswith("200")
        status_data = json.loads(body)
        if status_data["status"] == "done":
            assert "result" in status_data
            assert status_data["result"]["name"] == spec["name"]
            return
        if status_data["status"] == "failed":
            pytest.fail(f"Custom job failed: {status_data.get('error')}")

    pytest.fail("Custom queue job did not complete in time")
