#!/usr/bin/env python
"""
Client CLI for Customer Support Assistant

This script demonstrates how to use the ChatClient to communicate with the 
Orchestrator Agent from a command line interface.
"""

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
    Client to send tasks to the Orchestrator Agent and print its responses.
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


async def run_faq_query(chat_client):
    """Run an FAQ query through the chat client."""
    user_query = "What are your support hours?"
    
    console.rule("[bold blue]FAQ Query Example")
    console.print(f"[bold cyan]User Query:[/] {user_query}")
    
    # Get structured data first (if method is implemented)
    structured_data = {}
    try:
        extracted_data = await chat_client.extract_structured_data(user_query)
        if extracted_data:
            structured_data = extracted_data
    except (AttributeError, NotImplementedError):
        # Method might not be implemented yet
        pass
    
    try:
        # Send the query to the chat client
        task = await chat_client.chat(user_query, structured_data)
        
        console.rule("[bold green]FAQ Response")
        
        # Extract text content from the latest message
        text_content = []
        data_content = {}
        file_parts = []
        
        # Get the latest message from the task
        latest_message = None
        if task.status and task.status.message:
            latest_message = task.status.message
        
        # Process the message parts
        if latest_message and latest_message.parts:
            for part in latest_message.parts:
                if isinstance(part, TextPart):
                    text_content.append(part.text)
                elif isinstance(part, DataPart):
                    data_content.update(part.data)
                elif isinstance(part, FilePart) and part.file:
                    file_name = part.file.name or "file.dat"
                    mime_type = part.file.mimeType or "application/octet-stream"
                    file_parts.append((file_name, mime_type))
        
        # Display text response
        if text_content:
            console.print(f"[bold cyan]Text Response:[/] {''.join(text_content)}")
        else:
            console.print("[bold yellow]No text response received[/]")
        
        # Display structured data if available
        if data_content:
            console.print("[bold cyan]Structured Data:[/]")
            console.print(data_content)
        
        # Display file info if available
        if file_parts:
            console.print(f"[bold cyan]Files:[/] {len(file_parts)} files included")
            for filename, mime_type in file_parts:
                console.print(f"  - {filename} ({mime_type})")
                
        # Display task state
        if task.status and task.status.state:
            console.print(f"[bold cyan]Task State:[/] {task.status.state}")
            
    except Exception as e:
        logger.error(f"Error getting FAQ response: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/] {str(e)}")


async def run_technical_query(chat_client):
    """Run a technical query through the chat client."""
    user_query = "My internet connection is very slow and pages are not loading."
    data_content = {
        "device_type": "laptop",
        "os": "Windows 10",
        "connection_type": "WiFi"
    }
    
    console.rule("[bold blue]Technical Issue Query Example")
    console.print(f"[bold cyan]User Query:[/] {user_query}")
    console.print(f"[bold cyan]With Data:[/] {data_content}")
    
    try:
        # Send the query to the chat client
        task = await chat_client.chat(user_query, data_content)
        
        console.rule("[bold green]Technical Issue Response")
        
        # Extract text content from the latest message
        text_content = []
        response_data = {}
        file_parts = []
        
        # Get the latest message from the task
        latest_message = None
        if task.status and task.status.message:
            latest_message = task.status.message
            
        # Process the message parts
        if latest_message and latest_message.parts:
            for part in latest_message.parts:
                if isinstance(part, TextPart):
                    text_content.append(part.text)
                elif isinstance(part, DataPart):
                    response_data.update(part.data)
                elif isinstance(part, FilePart) and part.file:
                    file_name = part.file.name or "file.dat"
                    mime_type = part.file.mimeType or "application/octet-stream"
                    file_parts.append((file_name, mime_type))
        
        # Display text response
        if text_content:
            console.print(f"[bold cyan]Text Response:[/] {''.join(text_content)}")
        else:
            console.print("[bold yellow]No text response received[/]")
        
        # Display structured data if available
        if response_data:
            console.print("[bold cyan]Structured Data:[/]")
            console.print(response_data)
        
        # Display file info if available
        if file_parts:
            console.print(f"[bold cyan]Files:[/] {len(file_parts)} files included")
            for filename, mime_type in file_parts:
                console.print(f"  - {filename} ({mime_type})")
                
        # Display task state
        if task.status and task.status.state:
            console.print(f"[bold cyan]Task State:[/] {task.status.state}")
            
    except Exception as e:
        logger.error(f"Error getting technical response: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/] {str(e)}")


async def run_escalation_query(chat_client):
    """Run an escalation query through the chat client using streaming."""
    user_query = "I've tried everything, and I need to speak to a human agent now. My account number is 12345."
    data_content = {
        "user_id": "user123", 
        "account_number": "12345", 
        "previous_attempts": "Restarted modem, checked cables"
    }
    
    console.rule("[bold blue]Escalation Query Example (Streaming)")
    console.print(f"[bold cyan]User Query:[/] {user_query}")
    console.print(f"[bold cyan]With Data:[/] {data_content}")
    
    try:
        # Get streaming response for escalation
        console.print("[bold cyan]Streaming Response:[/]")
        
        # Keep track of accumulated content for final display
        accumulated_text = []
        accumulated_data = {}
        file_parts = []
        
        # Process streaming events
        async for event in chat_client.chat_streaming(user_query, data_content):
            # Handle TaskStatusUpdateEvent
            if isinstance(event, TaskStatusUpdateEvent):
                console.print(f"[dim]Status Update:[/] {event.status.state}")
                
                # Process message parts if available
                if event.status.message and event.status.message.parts:
                    for part in event.status.message.parts:
                        if isinstance(part, TextPart) and part.text:
                            accumulated_text.append(part.text)
                            console.print(f"[dim]Text Chunk:[/] {part.text}")
                            
                        elif isinstance(part, DataPart) and part.data:
                            accumulated_data.update(part.data)
                            console.print(f"[dim]Data Chunk:[/] {part.data}")
                            
                        elif isinstance(part, FilePart) and part.file:
                            file_name = part.file.name or "file.dat"
                            mime_type = part.file.mimeType or "application/octet-stream"
                            file_parts.append((file_name, mime_type))
                            console.print(f"[dim]File Received:[/] {file_name} ({mime_type})")
            
            # Handle TaskArtifactUpdateEvent
            elif isinstance(event, TaskArtifactUpdateEvent):
                console.print(f"[dim]Artifact Update:[/] {event.artifact.name}")
                
                # Process artifact parts if available
                if event.artifact.parts:
                    for part in event.artifact.parts:
                        if isinstance(part, TextPart) and part.text:
                            console.print(f"[dim]Artifact Text:[/] {part.text}")
                            
                        elif isinstance(part, DataPart) and part.data:
                            console.print(f"[dim]Artifact Data:[/] {part.data}")
                            
                        elif isinstance(part, FilePart) and part.file:
                            file_name = part.file.name or "artifact.dat"
                            mime_type = part.file.mimeType or "application/octet-stream"
                            console.print(f"[dim]Artifact File:[/] {file_name} ({mime_type})")
        
        # Display final accumulated response
        console.rule("[bold green]Complete Escalation Response")
        
        if accumulated_text:
            console.print(f"[bold cyan]Full Text Response:[/] {''.join(accumulated_text)}")
        else:
            console.print("[bold yellow]No text response received[/]")
        
        if accumulated_data:
            console.print("[bold cyan]Complete Structured Data:[/]")
            console.print(accumulated_data)
            
        if file_parts:
            console.print(f"[bold cyan]Files:[/] {len(file_parts)} files included")
            for filename, mime_type in file_parts:
                console.print(f"  - {filename} ({mime_type})")
                
    except Exception as e:
        logger.error(f"Error getting escalation response: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/] {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
