import os
import sys
import subprocess
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


class Handler(BaseHTTPRequestHandler):
    """Simple HTTP handler that serves the UI and runs the optimizer on demand."""

    def log_message(self, format, *args):
        # Suppress default access logs for cleaner output
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file("index.html", "text/html; charset=utf-8")
        elif parsed.path == "/run":
            self.run_optimizer()
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

    def run_optimizer(self):
        """Run main.py and return its console output as HTML."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        try:
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            output = result.stdout + (result.stderr or "")
        except subprocess.TimeoutExpired:
            output = "Optimization timed out after 5 minutes."
        except Exception as e:
            output = f"Error running optimizer: {e}"

        # Format the text output for safe HTML display
        escaped = html.escape(output)
        formatted = escaped.replace("\n", "<br>")
        self.wfile.write(formatted.encode("utf-8"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Server running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
