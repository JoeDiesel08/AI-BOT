import os
import sys
import subprocess
import threading
import json
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


class JobManager:
    """Runs the optimizer in the background and stores output for polling."""

    def __init__(self):
        self.lock = threading.Lock()
        self.output = []
        self.status = "idle"  # idle, running, completed, error
        self.process = None
        self.thread = None

    def start(self):
        with self.lock:
            if self.status == "running":
                return
            self.output = []
            self.status = "running"
            self.process = None
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        """Execute main.py and capture stdout/stderr line by line."""
        try:
            env = os.environ.copy()
            # The optimizer scale is controlled by environment variables on the host.
            # main.py defaults to 20 agents / 10 generations / 500 candles if unset.
            # Ensure child process flushes stdout line-by-line so progress is streamed live
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                [sys.executable, "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
            )
            for line in process.stdout:
                with self.lock:
                    self.output.append(line.rstrip("\n"))
            process.wait()
            with self.lock:
                if process.returncode != 0:
                    self.status = "error"
                else:
                    self.status = "completed"
        except Exception as e:
            with self.lock:
                self.output.append(f"Server error: {e}")
                self.status = "error"

    def snapshot(self, since=0):
        with self.lock:
            total = len(self.output)
            # Protect against the client sending a future index
            if since > total:
                since = total
            return {
                "status": self.status,
                "output": "\n".join(self.output[since:]),
                "total": total,
            }


job_manager = JobManager()


class Handler(BaseHTTPRequestHandler):
    """Simple HTTP handler that serves the UI and runs the optimizer on demand."""

    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file("index.html", "text/html; charset=utf-8")
        elif parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        elif parsed.path == "/start":
            job_manager.start()
            self._send_json({"status": "started"})
        elif parsed.path == "/progress":
            query = parse_qs(parsed.query)
            since = int(query.get("since", ["0"])[0])
            self._send_json(job_manager.snapshot(since=since))
        elif parsed.path == "/run":
            # Backwards-compatible single-request endpoint (streams when finished)
            job_manager.start()
            while job_manager.status == "running":
                threading.Event().wait(0.5)
            snapshot = job_manager.snapshot()
            escaped = html.escape(snapshot["output"])
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(escaped.replace("\n", "<br>").encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def serve_file(self, filename, content_type):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        except FileNotFoundError:
            self.send_error(404, "File Not Found")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Server running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
