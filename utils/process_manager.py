import subprocess
import time
import signal
import os
import requests
from threading import Thread
import atexit

class FastAPIServer:
    def __init__(self):
        # Initialize the server process as None
        self.server_process = None
        self.log_thread = None
        
        # Register cleanup function to ensure server is terminated when notebook kernel stops
        atexit.register(self.stop)
    
    def start(self, port=8000, app_module="your_fastapi_app:app", host="0.0.0.0"):
        """
        Start a FastAPI server in a subprocess
        
        Args:
            port (int): Port number to run the server on
            app_module (str): Module path to your FastAPI app (e.g., "main:app")
            host (str): Host to bind the server to
            
        Returns:
            subprocess.Popen: Process handle for the server
        """
        # If there's already a server running, stop it first
        if self.server_process is not None:
            self.stop()
        
        # Create a FastAPI server in a separate process
        cmd = ["uvicorn", app_module, f"--port={port}", f"--host={host}"]
        
        # Use start_new_session=True to create a new process group for proper termination
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True
        )
        
        print(f"FastAPI server starting on port {port}...")
        
        # Give the server a moment to start
        time.sleep(2)
        
        # Verify the server is running
        try:
            response = requests.get(f"http://localhost:{port}")
            print(f"Server is running! Status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("Server doesn't seem to be responding. Check the logs.")
        
        # Stream server logs to notebook
        self._start_log_streaming()
        
        return self.server_process
    
    def _start_log_streaming(self):
        """Start a thread to stream server logs to the notebook output"""
        def stream_logs():
            while self.server_process and self.server_process.poll() is None:
                output = self.server_process.stdout.readline()
                if output:
                    print(output.strip())
        
        # Start log streaming in a separate thread
        self.log_thread = Thread(target=stream_logs)
        self.log_thread.daemon = True  # Ensures the thread will die when notebook exits
        self.log_thread.start()
    
    def stop(self):
        """Stop the FastAPI server and clean up resources"""
        if self.server_process is None:
            print("No server is running.")
            return
        
        try:
            # Get the process group ID
            pgid = os.getpgid(self.server_process.pid)
            
            # Send SIGTERM to the entire process group
            os.killpg(pgid, signal.SIGTERM)
            
            # Give it a moment to shut down gracefully
            time.sleep(1)
            
            # Check if it's still running
            if self.server_process.poll() is None:
                print("Server still running, trying SIGKILL...")
                os.killpg(pgid, signal.SIGKILL)
            
            # Clean up
            self.server_process = None
            print("Server has been stopped.")
        except ProcessLookupError:
            # Process already gone
            self.server_process = None
            print("Server was already stopped.")
        except Exception as e:
            print(f"Error stopping server: {e}")
    
    def is_running(self):
        """Check if the server is currently running
        
        Returns:
            bool: True if server is running, False otherwise
        """
        if self.server_process is None:
            return False
        
        # poll() returns None if the process is still running
        return self.server_process.poll() is None
    
    def __del__(self):
        """Destructor to ensure server is stopped when object is deleted"""
        self.stop()