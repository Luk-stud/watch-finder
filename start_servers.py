import os
import sys
import subprocess
import webbrowser
from time import sleep

def start_backend():
    """Start the Flask backend server."""
    print("Starting backend server...")
    backend_path = os.path.join('backend', 'api', 'app.py')
    backend_process = subprocess.Popen([sys.executable, backend_path])
    return backend_process

def start_frontend():
    """Start the frontend server using Python's http.server."""
    print("Starting frontend server...")
    os.chdir('frontend')
    frontend_process = subprocess.Popen([sys.executable, '-m', 'http.server', '8080'])
    os.chdir('..')
    return frontend_process

def main():
    # Check if embeddings exist
    embeddings_path = os.path.join('embeddings', 'watch_image_embeddings.pkl')
    if not os.path.exists(embeddings_path):
        print("CLIP image embeddings not found. Generating embeddings...")
        subprocess.run([sys.executable, 'generate_clip_embeddings.py'])
    
    # Start servers
    backend_process = start_backend()
    frontend_process = start_frontend()
    
    # Wait for servers to start
    print("Waiting for servers to start...")
    sleep(2)
    
    # Open browser
    print("Opening browser...")
    webbrowser.open('http://localhost:8080')
    
    print("\nServers are running!")
    print("Frontend: http://localhost:8080")
    print("Backend: http://localhost:5001")
    print("Using CLIP image-based similarity matching!")
    print("\nPress Ctrl+C to stop the servers")
    
    try:
        # Keep the script running
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Servers stopped")

if __name__ == '__main__':
    main() 