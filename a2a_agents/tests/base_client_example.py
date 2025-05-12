import asyncio
import json
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import the base client
from base_a2a_client import BaseA2AClient

# Import helper functions
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from a2a_agents.a2a_utils.a2a_helper_functions import (
    create_task_params,
    extract_text_from_part,
    extract_data_from_part,
    extract_file_from_part
)

from common.types import (
    Message, TextPart, DataPart, Part,
    TaskSendParams,
    AgentCard
)

# Configuration
AGENT_BASE_URL = "http://localhost:8001"  # Agent server URL
OUTPUT_DIR = "client_output"  # Directory to save files
DEBUG = True  # Enable debug logging

# Example helper to print part contents
def print_part_contents(part: Part, label: str = "Part"):
    """Print the contents of a message part based on its type"""
    text_content = extract_text_from_part(part)
    data_content = extract_data_from_part(part)
    file_content = extract_file_from_part(part)
    
    if text_content:
        print(f"{label} Text: {text_content}")
    if data_content:
        print(f"{label} Data: {json.dumps(data_content, indent=2)}")
    if file_content:
        content, name, mime = file_content
        if isinstance(content, bytes):
            print(f"{label} File: {name} ({mime}) - {len(content)} bytes")
        else:  # URI
            print(f"{label} File URI: {content}")


async def run_standard_example():
    print("\n--- Running Standard Communication Example ---")
    
    # Get all agent cards from the server
    cards = await BaseA2AClient.get_agent_cards_from_url(AGENT_BASE_URL)
    print(f"Found {len(cards)} agent cards:")
    for i, card in enumerate(cards):
        print(f"{i+1}. {card.name} - {card.description}")
    
    if not cards:
        print("No agent cards found!")
        return
    
    # Create client using the first card
    client = BaseA2AClient(agent_card=cards[0], debug=DEBUG, output_dir=OUTPUT_DIR)
    
    # Generate a session ID for this conversation
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
      # Create a task with a simple text message
    message = BaseA2AClient.create_client_message(text_content="Hello! Please tell me about yourself.")
    task_params = create_task_params(message, session_id=session_id)
    
    # Send task and get response (standard communication)
    print("Sending task...")
    task_id, response = await client.send_task_standard(task_params)
    print(f"Task ID: {task_id}")
    
    # Display response
    print("\nAgent Response:")
    agent_messages = [msg for msg in client.get_task_messages(task_id) if msg.role == "agent"]
    
    if agent_messages:
        latest_message = agent_messages[-1]
        for i, part in enumerate(latest_message.parts):
            print_part_contents(part, f"Part {i+1}")
    else:
        print("No agent messages received")
    
    # Check if any files were saved
    files = client.get_task_files(task_id)
    if files:
        print("\nSaved files:")
        for file_path in files:
            print(f"- {file_path}")


async def run_streaming_example():
    print("\n--- Running Streaming Communication Example ---")
    
    # Get all agent cards from the server
    cards = await BaseA2AClient.get_agent_cards_from_url(AGENT_BASE_URL)
    if not cards:
        print("No agent cards found!")
        return
    
    # Create client using the first card
    client = BaseA2AClient(agent_card=cards[0], debug=DEBUG, output_dir=OUTPUT_DIR)
    
    # Generate a session ID for this conversation
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
      # Create a task with a simple text message
    message = BaseA2AClient.create_client_message(text_content="Please generate a sample image and tell me about it step by step.")
    task_params = create_task_params(message, session_id=session_id)
    
    # Send task and get streaming response
    print("Sending streaming task...")
    task_id, response_stream = await client.send_task_streaming(task_params)
    
    # Process streaming responses
    print("Receiving streaming responses...")
    print("-" * 50)
    
    try:
        async for response in response_stream:
            if hasattr(response.result, 'status'):
                status = response.result.status
                print(f"Status Update: {status.state}")
            
            elif hasattr(response.result, 'artifact'):
                artifact = response.result.artifact
                print(f"Artifact: {artifact.artifact_id} - {artifact.type}")
            
            elif hasattr(response.result, 'delta'):
                delta = response.result.delta
                if delta.role == "agent":
                    for part in delta.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"Agent: {part.text}", end='', flush=True)
                print()
    except Exception as e:
        print(f"Error during streaming: {e}")
    
    print("-" * 50)
    print("Streaming complete")
    
    # Show final state
    print("\nFinal task state:", client.get_current_status(task_id))
    
    # Check if any files were saved
    files = client.get_task_files(task_id)
    if files:
        print("\nSaved files:")
        for file_path in files:
            print(f"- {file_path}")


async def run_multi_modal_example():
    print("\n--- Running Multi-Modal Communication Example ---")
    
    # Get all agent cards from the server
    cards = await BaseA2AClient.get_agent_cards_from_url(AGENT_BASE_URL)
    if not cards:
        print("No agent cards found!")
        return
    
    # Create client using the first card
    client = BaseA2AClient(agent_card=cards[0], debug=DEBUG, output_dir=OUTPUT_DIR)
    
    # Generate a session ID for this conversation
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    
    # Create a temporary file with some content to send
    temp_file_path = os.path.join(OUTPUT_DIR, "example_data.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(temp_file_path, "w") as f:
        f.write("This is sample content to demonstrate file transfer capabilities.")
    
    print(f"Created temporary file: {temp_file_path}")
    
    # Create a message with text, data, and file
    message = BaseA2AClient.create_client_message(
        text_content=["Hello! This is a multi-modal message.", "Please analyze the following data and file."],
        data_content={
            "example_data": {
                "name": "Example",
                "values": [1, 2, 3, 4, 5],
                "metadata": {
                    "source": "BaseA2AClient example",
                    "timestamp": datetime.now().isoformat()
                }
            }
        },
        file_paths=[temp_file_path]
    )
    
    # Create task parameters
    task_params = create_task_params(message, session_id=session_id)
    
    # Send task and get response (standard communication for this example)
    print("Sending multi-modal task...")
    task_id, response = await client.send_task_standard(task_params)
    print(f"Task ID: {task_id}")
    
    # Display response
    print("\nAgent Response:")
    agent_messages = [msg for msg in client.get_task_messages(task_id) if msg.role == "agent"]
    
    if agent_messages:
        latest_message = agent_messages[-1]
        for i, part in enumerate(latest_message.parts):
            print_part_contents(part, f"Part {i+1}")
    else:
        print("No agent messages received")

    # Show task history information
    history = client.get_task_history(task_id)
    print(f"\nMessage count: {len(history['messages'])}")
    print(f"Current status: {history['current_status']}")
    
    # Check if any files were saved
    files = client.get_task_files(task_id)
    if files:
        print("\nSaved files:")
        for file_path in files:
            print(f"- {file_path}")


async def main():
    print("=== BaseA2AClient Example ===")
    
    try:
        # Run standard communication example
        await run_standard_example()
        
        # Run streaming communication example
        await run_streaming_example()
        
        # Run multi-modal example
        await run_multi_modal_example()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
