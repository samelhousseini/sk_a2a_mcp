import asyncio
import json
import uuid
from typing import cast, Dict, Any

# Adjust path to import from the common module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'python')))

# Import helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from a2a_agents.a2a_utils.a2a_helper_functions import (
    create_task_params,
    process_standard_response,
    handle_client_error
)

# Import our BaseA2AClient
from agents_examples.base_a2a_client import BaseA2AClient

from common.types import (
    Message, TextPart, DataPart,
    TaskSendParams,
    AgentCard,
    SendTaskResponse, Task,
    JSONRPCError,
    A2AClientHTTPError, A2AClientJSONError
)

AGENT_BASE_URL = "http://localhost:8001" # Agent server URL
OUTPUT_DIR = "./downloaded_files"  # Directory for downloaded files
DEBUG = True  # Enable debug logging

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

    # Step 3: Use create_client_message method to create message with data
    client_message = BaseA2AClient.create_client_message(
        data_content={
            "amount": 200,
            "from_currency": "EUR",
            "to_currency": "JPY"
        }
    )

    # Use helper function to create task parameters
    task_send_params = create_task_params(
        message=client_message,
        task_id=uuid.uuid4().hex,
        session_id=session_id,
        metadata={"client_session_id": session_id}
    )
    print(f"--- Calling send_task_standard via BaseA2AClient for session: {session_id} ---")
    print(f"Task parameters:\n{json.dumps(task_send_params.model_dump(exclude_none=True), indent=2)}\n")
    
    try:
        # Step 4: Call send_task_standard and process response
        task_id, response = await a2a_client.send_task_standard(task_send_params)
        
        # Process the response using the helper function
        result = process_standard_response(
            client=a2a_client, 
            task_id=task_id, 
            response=response, 
            print_output=True
        )

    except A2AClientHTTPError as e:
        handle_client_error(e, "BaseA2AClient HTTP")
    except A2AClientJSONError as e:
        handle_client_error(e, "BaseA2AClient JSON")
    except Exception as e:
        handle_client_error(e, "Unexpected")

if __name__ == "__main__":
    asyncio.run(run_client())
