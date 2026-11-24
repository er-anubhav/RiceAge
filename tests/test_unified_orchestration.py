import os
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import pytest

from src.rice_age_client import RiceAgeClient


class MockRPiServer(BaseHTTPRequestHandler):
    active_session = {}
    is_busy = False

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if self.path == "/api/session/set":
            MockRPiServer.active_session = data
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "session": data}).encode("utf-8"))
            return

        elif self.path == "/api/test/start":
            if MockRPiServer.is_busy:
                self.send_response(409)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "busy", "message": "Test in progress"}).encode("utf-8"))
                return
            MockRPiServer.is_busy = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "estimated_duration_sec": 21}).encode("utf-8"))
            return

        elif self.path == "/api/test/abort":
            MockRPiServer.is_busy = False
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "aborting"}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/test/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "state": "TESTING" if MockRPiServer.is_busy else "IDLE",
                "live_step": "Awaiting Color Reaction" if MockRPiServer.is_busy else "Ready",
                "progress": 65 if MockRPiServer.is_busy else 100,
                "step_remaining": 8,
                "batch_no": MockRPiServer.active_session.get("batch_no", "BATCH-001"),
                "lot_no": MockRPiServer.active_session.get("lot_no", "LOT-A")
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif self.path == "/api/camera/preview":
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            # Fake tiny JPEG byte sequence
            self.wfile.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00")
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress server logs during test


@pytest.fixture(scope="module")
def mock_rpi_tester():
    server = HTTPServer(("127.0.0.1", 0), MockRPiServer)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield ("127.0.0.1", port)
    server.shutdown()


def test_set_active_session(tmp_path, mock_rpi_tester):
    host, port = mock_rpi_tester
    cache_file = str(tmp_path / "cache.json")
    client = RiceAgeClient(mdns_host=host, port=port, uploads_dir=str(tmp_path), cache_file=cache_file)
    client._cached_ip = host

    ok = client.set_active_session("Sample-0042", "BATCH-2026-X", "LOT-GOLD", "Raw Rice")
    assert ok is True
    assert MockRPiServer.active_session.get("batch_no") == "BATCH-2026-X"
    assert MockRPiServer.active_session.get("lot_no") == "LOT-GOLD"


def test_trigger_remote_test_and_status(tmp_path, mock_rpi_tester):
    host, port = mock_rpi_tester
    cache_file = str(tmp_path / "cache.json")
    client = RiceAgeClient(mdns_host=host, port=port, uploads_dir=str(tmp_path), cache_file=cache_file)
    client._cached_ip = host

    # Reset server state
    MockRPiServer.is_busy = False

    # Trigger test
    res = client.trigger_remote_test("Sample-0042", "BATCH-2026-X", "LOT-GOLD")
    assert res.get("status") == "started"
    assert res.get("estimated_duration_sec") == 21

    # Check status during test
    stat = client.get_detailed_status()
    assert stat.get("state") == "TESTING"
    assert stat.get("progress") == 65
    assert stat.get("connected") is True

    # Check 409 conflict when busy
    res_busy = client.trigger_remote_test("Sample-0043")
    assert res_busy.get("status") == "busy"

    # Abort test
    aborted = client.abort_remote_test()
    assert aborted is True
    assert MockRPiServer.is_busy is False


def test_get_chamber_preview(tmp_path, mock_rpi_tester):
    host, port = mock_rpi_tester
    cache_file = str(tmp_path / "cache.json")
    client = RiceAgeClient(mdns_host=host, port=port, uploads_dir=str(tmp_path), cache_file=cache_file)
    client._cached_ip = host

    jpeg_bytes = client.get_chamber_preview()
    assert jpeg_bytes is not None
    assert len(jpeg_bytes) > 10
    assert jpeg_bytes.startswith(b"\xff\xd8")  # JPEG SOI marker
