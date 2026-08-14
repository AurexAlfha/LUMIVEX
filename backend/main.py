"""
LUMIVEX Backend
Version: 0.1.0

This is the initial backend foundation.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json


HOST = "0.0.0.0"
PORT = 8000


class LumivexHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()

        self.wfile.write(response)

    def do_GET(self):
        if self.path == "/":
            self.send_json({
                "name": "LUMIVEX",
                "status": "online",
                "version": "0.1.0"
            })

        elif self.path == "/health":
            self.send_json({
                "status": "healthy"
            })

        else:
            self.send_json({
                "error": "Endpoint not found"
            }, 404)


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), LumivexHandler)

    print("LUMIVEX backend starting...")
    print(f"Server running on port {PORT}")

    server.serve_forever()
