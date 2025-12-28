#!/usr/bin/env python3
"""
Simple HTTP server to serve the AI Engineering Course website locally
Run this script and open http://localhost:8000 in your browser
"""

import http.server
import socketserver
import os
import sys

# Change to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🚀 AI Engineering Course website running at:")
        print(f"   http://localhost:{PORT}")
        print(f"   http://127.0.0.1:{PORT}")
        print()
        print("📂 Available pages:")
        print(f"   Main Course: http://localhost:{PORT}")
        print(f"   Week 1: http://localhost:{PORT}/weeks/week1/")
        print(f"   Session 1.1: http://localhost:{PORT}/weeks/week1/session1-1/topic1.html")
        print()
        print("Press Ctrl+C to stop the server")
        print("-" * 60)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")

if __name__ == "__main__":
    main()