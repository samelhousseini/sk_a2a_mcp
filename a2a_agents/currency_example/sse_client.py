import asyncio
import json
import uuid
import os
from typing import cast

# Adjust path to import from the common module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'python')))

# Import helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from a2a_agents.a2a_utils.a2a_helper_functions import (
    create_task_params,
    process_streaming_response,
    print_task_summary,
    handle_client_error,
    print_part_contents  # Our new helper function
)

# Import our BaseA2AClient
from agents_examples.base_a2a_client import BaseA2AClient

from common.types import (
    Message, TextPart, DataPart, Part,
    TaskSendParams,
    AgentCard,
    SendTaskStreamingResponse, 
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent,
    JSONRPCError,
    A2AClientHTTPError, A2AClientJSONError
)

AGENT_BASE_URL = "http://localhost:8001" # Agent server URL
OUTPUT_DIR = "./downloaded_files"  # Directory for downloaded files
DEBUG = True  # Enable debug logging

# The helper function for printing part contents has been moved to a2a_helper_functions.py

async def run_client():
    session_id = str(uuid.uuid4())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Resolving Agent Card from {AGENT_BASE_URL} ---")
    try:
        # Step 1: Get agent cards using the static method from BaseA2AClient
        cards = await BaseA2AClient.get_agent_cards_from_url(AGENT_BASE_URL)
        if not cards:
            print("No agent cards found!")
            return
        
        agent_card = cards[0]  # Use the first card
        print(f"Successfully resolved agent card: {agent_card.name}")
        
        # Step 2: Instantiate BaseA2AClient with the resolved card
        a2a_client = BaseA2AClient(agent_card=agent_card, output_dir=OUTPUT_DIR, debug=DEBUG)
    except Exception as e:
        print(f"Error resolving agent card: {e}")
        print("Falling back to instantiating BaseA2AClient with URL directly.")
        a2a_client = BaseA2AClient(url=AGENT_BASE_URL, output_dir=OUTPUT_DIR, debug=DEBUG)

    # Step 3: Create a message with data using the BaseA2AClient method
    client_message = BaseA2AClient.create_client_message(
        data_content={
            "amount": 300,
            "from_currency": "USD",
            "to_currency": "GBP"
        }
    )    # Use helper function to create task parameters
    task_send_params = create_task_params(
        message=client_message,
        task_id=uuid.uuid4().hex,
        session_id=session_id,
        metadata={"client_session_id": session_id}
    )

    print(f"--- Calling send_task_streaming via BaseA2AClient for session: {session_id} ---")
    print(f"Task parameters:\n{json.dumps(task_send_params.model_dump(exclude_none=True), indent=2)}\n")

    try:
        # Step 4: Call send_task_streaming using our enhanced client
        # This returns a tuple with the task_id and an async generator
        task_id, response_stream = await a2a_client.send_task_streaming(task_send_params)
        print(f"Initial task ID: {task_id} (may be updated once stream begins)")
        
        # Process streaming responses
        async for sse_response in response_stream:            # Use our helper function to process and display the response (using synchronous function)
            result = process_streaming_response(
                response=sse_response,
                print_output=True,
                indent="  "
            )
            
            # Handle specific event types with immediate flushing
            if result["type"] == "status" and "state" in result["content"]:
                print(f"Status: {result['content']['state']}", flush=True)
                
            # For text content, display immediately with proper flushing
            if sse_response.result and hasattr(sse_response.result, 'status') and hasattr(sse_response.result.status, 'message'):
                message = sse_response.result.status.message
                if message and hasattr(message, 'parts'):
                    for part in message.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"└─ {part.text}", flush=True)
            
            # Ensure output is flushed immediately for real-time display
            sys.stdout.flush()
            
            # Check if there was an error and stop processing if needed
            if result["error"]:
                print("--- Error received from agent, stopping. ---")
                break # Stop processing on error
                
        print("--- SSE stream finished or iteration completed. ---")
        
        # Retrieve and print final task information using our helper function
        real_task_id = task_id if task_id != "pending" else None
        if real_task_id:
            print_task_summary(a2a_client, real_task_id)

    except A2AClientHTTPError as e:
        handle_client_error(e, "BaseA2AClient HTTP")
    except A2AClientJSONError as e:
        handle_client_error(e, "BaseA2AClient JSON")
    except Exception as e:
        handle_client_error(e, "Unexpected")

if __name__ == "__main__":
    asyncio.run(run_client())
