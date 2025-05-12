import asyncio
import subprocess
import time
import os
import sys
import signal
import platform

"""
This script tests the updated currency agent and clients by:
1. Starting the currency agent in the background
2. Running the standard client to test non-streaming communication
3. Running the SSE client to test streaming communication
"""

# Getting the path to the currency agent and client scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_script = os.path.join(current_dir, "currency_agent.py")
standard_client_script = os.path.join(current_dir, "standard_client.py")
sse_client_script = os.path.join(current_dir, "sse_client.py")

async def run_test():
    print(f"\n{'='*80}\nStarting Currency Agent Test\n{'='*80}")
    
    # Start the agent server as a subprocess
    print("\nStarting currency agent server...")
    agent_process = subprocess.Popen(
        [sys.executable, agent_script],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server to start
    print("Waiting for server to initialize (5 seconds)...")
    time.sleep(5)
    
    try:
        # Test the standard client
        print("\n\n" + "="*40)
        print("Running standard client test...")
        print("="*40)
        standard_result = subprocess.run(
            [sys.executable, standard_client_script],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        print("\nStandard Client Output:")
        print(standard_result.stdout)
        if standard_result.stderr:
            print("Standard Client Errors:")
            print(standard_result.stderr)
            
        # Test the SSE client
        print("\n\n" + "="*40)
        print("Running SSE client test...")
        print("="*40)
        sse_result = subprocess.run(
            [sys.executable, sse_client_script],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        print("\nSSE Client Output:")
        print(sse_result.stdout)
        if sse_result.stderr:
            print("SSE Client Errors:")
            print(sse_result.stderr)
            
    finally:
        # Terminate the agent server
        print("\n\n" + "="*40)
        print("Stopping currency agent server...")
        
        if platform.system() == "Windows":
            # On Windows, we need to use taskkill to terminate the process tree
            subprocess.run(f"taskkill /F /T /PID {agent_process.pid}", 
                          shell=True, 
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
        else:
            # On Unix-like systems, we can send a signal
            os.killpg(os.getpgid(agent_process.pid), signal.SIGTERM)
            agent_process.terminate()
        
        # Get any output from the agent
        stdout, stderr = agent_process.communicate(timeout=5)
        print("\nAgent Server Output:")
        print(stdout)
        if stderr:
            print("Agent Server Errors:")
            print(stderr)
            
    print(f"\n{'='*80}\nTest Complete\n{'='*80}")

if __name__ == "__main__":
    asyncio.run(run_test())
