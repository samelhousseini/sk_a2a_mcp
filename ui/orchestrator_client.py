import asyncio
import logging
import uuid
import os
import sys
from rich.console import Console 
console = Console()

# Add parent directory to path so we can import a2a_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from a2a_agents.base_a2a_client import BaseA2AClient
    from a2a_agents.a2a_utils.a2a_helper_functions import (
        create_task_params, create_client_message, extract_message_content,
        extract_user_query, create_agent_response, log_agent_response, 
        process_streaming_response, extract_text_from_part
    )
    from common.types import (
        Message, TextPart, DataPart, FilePart, Part, 
        TaskStatusUpdateEvent, TaskArtifactUpdateEvent
    )
    # Import the ChatClient class
    from ui.app_client import ChatClient
except ImportError as e:
    logging.error(f"Import error: {e}. Make sure the repository structure is correct.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ORCHESTRATOR_AGENT_URL = os.environ.get("ORCHESTRATOR_AGENT_URL", "http://localhost:8000")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def main():
    """
    Client to send tasks to the OrchestratorAgent and print its responses.
    This version uses the ChatClient class that's also used by the Chainlit UI.
    """
    # Create chat client instance
    chat_client = ChatClient(url=ORCHESTRATOR_AGENT_URL, output_dir=OUTPUT_DIR)
    logger.info(f"Chat client initialized. Target URL: {ORCHESTRATOR_AGENT_URL}")

    # --- Example 1: FAQ Query ---
    await run_faq_query(chat_client)

    # --- Example 2: Technical Issue Query ---
    await run_technical_query(chat_client)

    # --- Example 3: Escalation Query ---
    await run_escalation_query(chat_client)


async def run_faq_query(client):
    """Run an FAQ query through the orchestrator agent."""
    user_query = "What are your support hours?"
    
    logger.info(f"\n--- Sending FAQ Query to Orchestrator ---")
    logger.info(f"User Query: {user_query}")

    # Get response using the client
    task = await client.chat(user_query)
    
    # Process and display the task response
    logger.info("Orchestrator Response:")
    
    # Get the latest message
    latest_message = None
    if task.status and task.status.message:
        latest_message = task.status.message
    
    # Extract and display message content
    if latest_message and latest_message.parts:
        for part in latest_message.parts:
            if isinstance(part, TextPart):
                logger.info(f"Text: {part.text}")
            elif isinstance(part, DataPart):
                logger.info(f"Data: {part.data}")
            elif isinstance(part, FilePart):
                logger.info(f"File: {part.file.name} ({part.file.mime_type})")


async def run_technical_query(client):
    """Run a technical troubleshooting query through the orchestrator agent."""
    user_query = "My internet connection is very slow and pages are not loading."

    logger.info(f"\n--- Sending Technical Issue Query to Orchestrator ---")
    logger.info(f"User Query: {user_query}")

    # Get response using the client with some technical data
    task = await client.chat(
        user_query,
        data_content={
            "device_type": "laptop",
            "os": "Windows 10",
            "connection_type": "WiFi"
        }
    )
    
    # Process and display the task response
    logger.info("Orchestrator Response:")
    
    # Get the latest message
    latest_message = None
    if task.status and task.status.message:
        latest_message = task.status.message
    
    # Extract and display message content
    if latest_message and latest_message.parts:
        for part in latest_message.parts:
            if isinstance(part, TextPart):
                logger.info(f"Text: {part.text}")
            elif isinstance(part, DataPart):
                logger.info(f"Data: {part.data}")
            elif isinstance(part, FilePart):
                logger.info(f"File: {part.file.name} ({part.file.mime_type})")


async def run_escalation_query(client):
    """Run a human escalation query through the orchestrator agent."""
    user_query = "I've tried everything, and I need to speak to a human agent now. My account number is 12345."

    logger.info(f"\n--- Sending Escalation Query to Orchestrator (Streaming) ---")
    logger.info(f"User Query: {user_query}")
    
    # Get streaming response for escalation
    logger.info("Streaming response:")
    
    # Process and display the streaming response
    async for event in client.chat_streaming(
        user_query,
        data_content={
            "user_id": "user123", 
            "account_number": "12345", 
            "previous_attempts": "Restarted modem, checked cables"
        }
    ):
        if isinstance(event, TaskStatusUpdateEvent):
            if event.status and event.status.state:
                logger.info(f"Status: {event.status.state}")
            
            if event.status and event.status.message and event.status.message.parts:
                for part in event.status.message.parts:
                    if isinstance(part, TextPart):
                        logger.info(f"Text: {part.text}")
                    elif isinstance(part, DataPart):
                        logger.info(f"Data: {part.data}")
                    elif isinstance(part, FilePart):
                        logger.info(f"File: {part.file.name} ({part.file.mime_type})")
        
        elif isinstance(event, TaskArtifactUpdateEvent) and event.artifact:
            logger.info(f"Artifact: {event.artifact.name} ({event.artifact.mime_type})")


if __name__ == "__main__":
    asyncio.run(main())
