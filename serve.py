#!/usr/bin/env python3
"""
Simple HTTP server to serve the Bloch-McConnell HTML page on port 8001.
"""

import http.server
import socketserver
import os

PORT = 8002

# Change to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving Bloch-McConnell simulation at http://localhost:{PORT}/bloch_mcconnell.html")
    print(f"Or visit http://localhost:{PORT}/ to see all files")
    print("Press Ctrl+C to stop the server")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

