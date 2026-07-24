import os
import sys
import subprocess
import threading
import json
import html
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/kraken_paper"))



MODE = os.environ.get("MODE", "optimize").lower()


def _start_trading_bot():
    """Launch the live/paper trading bot in a daemon thread at startup."""
    try:
        from trading_bot import TradingBot
        if MODE == "live":
            os.environ["PAPER_TRADING"] = "false"
        bot = TradingBot()
        thread = threading.Thread(target=bot.run, daemon=True)
        thread.start()
        print(f"Started {MODE.upper()} trading bot thread.")
        return bot
    except Exception as e:
        print(f"Could not start trading bot: {e}")


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


def _read_tail_jsonl(path: Path, max_lines=100):
    """Return the last N JSON objects from a JSONL file, oldest first."""
    try:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        records = []
        for line in lines[-max_lines:]:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
        return records
    except Exception:
        return []


def _get_state():
    """Read the paper/live trading state files from the persistent volume."""
    portfolio = {}
    try:
        pf = DATA_DIR / "portfolio.json"
        if pf.exists():
            with open(pf, "r", encoding="utf-8") as f:
                portfolio = json.load(f)
    except Exception as e:
        portfolio = {"error": str(e)}

    return {
        "portfolio": portfolio,
        "trades": _read_tail_jsonl(DATA_DIR / "trades.jsonl", 20),
        "equity": _read_tail_jsonl(DATA_DIR / "equity_log.jsonl", 200),
    }


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
            if MODE in ("paper", "live"):
                self.serve_file("dashboard.html", "text/html; charset=utf-8")
            else:
                self.serve_file("index.html", "text/html; charset=utf-8")
        elif parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        elif parsed.path == "/state":
            self._send_json(_get_state())
        elif parsed.path == "/mode":
            self._send_json({
                "mode": MODE,
                "pair": os.environ.get("TRADING_PAIR", "BTC/USD"),
                "paper": os.environ.get("PAPER_TRADING", "true").lower() in ("1", "true", "yes"),
                "validate": os.environ.get("KRAKEN_VALIDATE", "true").lower() in ("1", "true", "yes"),
            })
        elif parsed.path == "/start":
            if MODE in ("paper", "live"):
                if _trading_bot is None:
                    self._send_json({"status": "error", "message": "Trading bot not initialized"}, status=500)
                else:
                    threading.Thread(target=_trading_bot.run_once, daemon=True).start()
                    self._send_json({"status": "trade_iteration_triggered"})
            else:
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


_trading_bot = None

if __name__ == "__main__":
    if MODE in ("paper", "live"):
        _trading_bot = _start_trading_bot()

    port = int(os.environ.get("PORT", 8080))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Server running at http://localhost:{port} (MODE={MODE})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
