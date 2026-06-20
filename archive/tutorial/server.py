import http.server
import socketserver
import webbrowser
import os

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print("\n" + "="*50)
        print("🚀 IntelligentGaze Local Server Started!")
        print("="*50)
        print(f"\n👉 Step 1: Open your browser and go to:")
        print(f"   {url}")
        print(f"\n👉 Step 2: Allow Camera access to start eye-tracking.")
        print(f"\n👉 Step 3: Follow the on-screen instructions to collect data.")
        print("\n" + "="*50)
        print("(Press Ctrl+C to stop the server)")
        
        # Optional: Automatically open the browser
        # webbrowser.open(url)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")

if __name__ == "__main__":
    start_server()
