import os
import asyncio
import uuid
import json
import sys
import base64
import re
import time
from io import BytesIO
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
import copy
# Load environment variables
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.traceback import install as install_rich_traceback
console = Console()
install_rich_traceback(show_locals=True)

# Add a divider function to make output more readable
def print_divider(title=""):
    console.print(f"[bold yellow]{'=' * 30} {title} {'=' * 30}[/]")

import chainlit as cl
from chainlit import run_sync, make_async

# Add parent directory to path to import needed modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our client
from ui.app_client import ChatClient
from common.types import (
    TaskState, TaskStatusUpdateEvent, TaskArtifactUpdateEvent,
    TextPart, DataPart, FilePart
)

from a2a_agents.a2a_utils.a2a_helper_functions import (
    # Status and artifact functions
    update_task_status, 
    add_task_artifact,
    send_status_update_event,
    add_task_artifact_event,
    # Part creation functions
    create_text_part,
    create_data_part,
    create_message,
    # Part extraction functions
    parse_message_parts,
    extract_text_from_part,
    extract_data_from_part,
    extract_all_data_parts,
    extract_all_text_parts,
    extract_all_file_parts,
)



# Initialize client
chat_client = ChatClient()


# Store user sessions
user_sessions = {}

@cl.on_chat_start
async def start():
    """Initialize chat session when a user starts a new conversation"""
    
    print_divider("NEW CHAT SESSION")
    console.print("[bold green]Initializing new chat session...[/]")
    
    # Create a welcome message
    welcome_message = (
        "👋 Welcome to Customer Support! I'm your virtual assistant.\n\n"
        "I can help you with:\n"
        "- Answering common questions\n"
        "- Technical troubleshooting\n"
        "- Connecting you to human support\n\n"
        "How can I help you today?"
    )
    
    # Display the welcome message in the console
    console.print(Panel(welcome_message, title="Welcome Message", border_style="blue"))
    
    await cl.Message(content=welcome_message).send()
    
    # Create a new session for the user
    user_id = cl.user_session.get("user_id")
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        cl.user_session.set("user_id", user_id)
        console.print(f"[yellow]Created new user ID:[/] {user_id}")
    else:
        console.print(f"[yellow]Existing user ID:[/] {user_id}")
    
    # Store any session information
    user_sessions[user_id] = {
        "history": [],
        "current_issue": None
    }
    console.print("[green]Session initialized successfully[/]")


@cl.on_message
async def main(message: cl.Message):
    """Process incoming user messages"""
      # Get user query
    console.print(f"[bold blue]User message:[/] {message.content}")
    user_query = message.content.strip()
    
    # Get user ID from session
    user_id = cl.user_session.get("user_id")
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        cl.user_session.set("user_id", user_id)
    
    console.print(f"[dim]Processing request for User ID:[/] {user_id}")
    
    # Prepare data context to send with the query
    data_context = {
        "user_id": user_id,
    }
    
    # Add current issue to context if available
    if user_id in user_sessions and "current_issue" in user_sessions[user_id]:
        current_issue = user_sessions[user_id].get("current_issue")
        if current_issue:
            data_context["current_issue_key"] = current_issue
    
    # Create UI elements list for displaying files and special content
    elements = []
    
    console.print("[bold green]Starting streaming response[/]", style="bold")
    console.print("─" * 50)
    
    # For tracking accumulated content
    message_content = []
    accumulated_files = []
    accumulated_data = {}
      # Start streaming the response
    async with cl.Step(name="Processing") as step:
        try:
            # Stream the response from the chat client
            console.print("[yellow]Starting response stream...[/]")
            async for event in chat_client.chat_streaming(user_query, data_context):
                
                # Process TaskStatusUpdateEvent
                if isinstance(event, TaskStatusUpdateEvent) and event.status and event.status.message:
                    message = event.status.message
                    
                    # Log the event type and status
                    console.print(f"[yellow]Event:[/] TaskStatusUpdateEvent [dim]State: {event.status.state}[/]")
                    
                    # Process the message parts
                    if message.parts:
                        for part in message.parts:                            
                            # Handle text parts
                            if isinstance(part, TextPart) and part.text:
                                message_content.append(part.text)
                                # Display text chunks with rich formatting
                                console.print(f"[cyan]Text chunk:[/] {part.text[:50]}{'...' if len(part.text) > 50 else ''}")
                                # Update content directly on the message object and then call update()
                                if event.status.state == TaskState.COMPLETED:
                                    # Show that the assistant is thinking
                                    assistant_msg = cl.Message(content="", author="Assistant")
                                    await assistant_msg.send()
                                    assistant_msg.content = part.text
                                    await assistant_msg.update()
                                  # Handle data parts
                            elif isinstance(part, DataPart) and part.data:
                                accumulated_data.update(part.data)
                                
                                # Log the data with rich formatting
                                console.print("[magenta]Data part:[/]")
                                console.print(part.data, style="italic")
                                
                                # Add a system message if we get important data
                                for key in ["case_id", "issue_type", "escalation_status"]:
                                    if key in part.data and part.data[key]:
                                        await step.stream_token(f"\nReceived data: {key}={part.data[key]}")                              # Handle file parts
                            elif isinstance(part, FilePart) and part.file:
                                if hasattr(part.file, 'bytes') and part.file.bytes:
                                    # Log file details
                                    console.print(f"[green]File part:[/] {part.file.name or 'unnamed'} [dim]({part.file.mimeType or 'unknown type'})[/]")
                                    
                                    # Decode base64 data if present
                                    file_bytes = base64.b64decode(part.file.bytes) if part.file.bytes else b''
                                    file_size = len(file_bytes)
                                    console.print(f"[dim]File size:[/] {file_size:,} bytes")
                                    
                                    # Create a chainlit file element
                                    file_element = cl.File(
                                        name=part.file.name or "file.txt",
                                        mime=part.file.mimeType or "application/octet-stream",
                                        content=file_bytes
                                    )
                                    
                                    accumulated_files.append(file_element)
                                    await step.stream_token(f"\nReceived file: {part.file.name}")                    # Update task state in UI
                    if event.status.state:
                        state_name = event.status.state.name.lower()
                        text_contents = "\n".join(extract_all_text_parts(message))
                        await step.stream_token(f"\nStatus update: {state_name} - {text_contents}")
                        
                        # Log state change with nice formatting
                        console.print(f"[bold yellow]State change:[/] {state_name}", style="reverse")
                        
                        # Add system messages for important state changes
                        # if state_name in ["running", "completed", "failed"]:
                        #     status_text = {
                        #         "running": "Working on your request...",
                        #         "completed": "Request completed successfully",
                        #         "failed": "Request failed. Please try again."
                        #     }.get(state_name, state_name)
                            
                        #     await cl.Message(
                        #         content=status_text,
                        #         author="System"
                        #     ).send()
                  # Process TaskArtifactUpdateEvent
                elif isinstance(event, TaskArtifactUpdateEvent) and event.artifact:
                    artifact = event.artifact
                    artifact_name = artifact.name or "Generated content"
                    
                    # Log artifact event with pretty formatting
                    console.print(f"[bold blue]Artifact received:[/] {artifact_name}", style="on black")
                    if hasattr(artifact, 'description') and artifact.description:
                        console.print(f"[blue]Description:[/] {artifact.description}")
                    
                    await step.stream_token(f"\nArtifact received: {artifact_name}")
                      # Notify the user about the artifact (if it contains downloadable content)
                    if hasattr(artifact, 'parts'):
                        for part in artifact.parts:                            
                            if isinstance(part, FilePart) and part.file and part.file.bytes:
                                try:
                                    file_bytes = base64.b64decode(part.file.bytes)
                                    file_size = len(file_bytes)
                                    console.print(f"[green]Artifact file:[/] {part.file.name or artifact_name}")
                                    console.print(f"[dim]Size:[/] {file_size:,} bytes")
                                    
                                    artifact_file = cl.File(
                                        name=part.file.name or artifact_name,
                                        mime=part.file.mimeType or "application/octet-stream",
                                        content=file_bytes
                                    )
                                    accumulated_files.append(artifact_file)
                                    
                                    await cl.Message(
                                        content=f"I've prepared a resource for you: {part.file.name or artifact_name}",
                                        author="System",
                                        elements=[artifact_file]
                                    ).send()
                                except Exception as e:
                                    await step.stream_token(f"\nError processing artifact file: {str(e)}")
                    # Final update with all accumulated files and content
            if accumulated_files:
                # Update content directly on the message object first
                assistant_msg.content = "".join(message_content)
                # Then update with elements
                assistant_msg.elements = accumulated_files
                await assistant_msg.update()
            
            # Process any special data at the end
            response_text = "".join(message_content)
            
            # Display case ID if found
            case_id = accumulated_data.get("case_id")
            if case_id:
                case_element = cl.Text(
                    name="Case ID",
                    content=case_id,
                    display="inline",
                )
                await cl.Message(
                    content="Your case has been recorded in our system.",
                    author="System",
                    elements=[case_element]
                ).send()
            
            # Update the user session with issue type if found
            issue_type = accumulated_data.get("issue_type")
            if issue_type and user_id in user_sessions:
                user_sessions[user_id]["current_issue"] = issue_type
        except Exception as e:
            # Handle errors gracefully
            console.print(f"[bold red]Error:[/] {str(e)}", style="bold on white")
            console.print_exception(show_locals=True)
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            await cl.Message(content=error_message).send()
            
    # Store the interaction in history
    if user_id in user_sessions:
        user_sessions[user_id]["history"].append({
            "user": user_query,
            "assistant": "".join(message_content),
            "data": accumulated_data,
            "files": [f.name for f in accumulated_files] if accumulated_files else []
        })
        
        # Log the interaction history
        console.print("[bold blue]Interaction stored in history[/]")
        console.print(f"[dim]History size:[/] {len(user_sessions[user_id]['history'])} interactions")
        
    print_divider("END OF PROCESSING")
    
    # Additional statistics for console output
    console.print(Panel(
        f"[bold]Request Statistics[/]\n"
        f"Text chunks: {len(message_content)}\n"
        f"Files: {len(accumulated_files)}\n"
        f"Data keys: {', '.join(accumulated_data.keys()) if accumulated_data else 'None'}\n",
        title="Response Summary", 
        border_style="green"
    ))


@cl.on_settings_update
async def setup_agent(settings):
    """Update agent settings when the user changes them"""
    console.print(f"[bold magenta]Settings updated:[/]")
    console.print(settings, style="dim")


# Log the state changes during processing
def log_state_change(state_name, event_type="State Change"):
    state_colors = {
        "running": "yellow",
        "completed": "green",
        "failed": "red",
        "pending": "blue",
        "cancelled": "red"
    }
    color = state_colors.get(state_name.lower(), "white")
    console.print(f"[bold {color}]{event_type}:[/] {state_name}", style="reverse")


# Utility function to print artifact details
def log_artifact(name, description=None, size=None):
    """Print artifact details with rich formatting"""
    details = []
    if description:
        details.append(f"[dim]Description:[/] {description}")
    if size:
        details.append(f"[dim]Size:[/] {size:,} bytes")
        
    console.print(f"[bold blue]Artifact:[/] {name}", style="on black")
    for detail in details:
        console.print(f"  {detail}")
        
# Utility function to summarize processed message parts
# def log_message_stats():
#     """Print statistics about processed message parts"""
#     console.print(Panel(
#         f"[bold]Message Statistics[/]\n"
#         f"Text chunks: {len(message_content)}\n"
#         f"Files: {len(accumulated_files)}\n"
#         f"Data keys: {', '.join(accumulated_data.keys()) if accumulated_data else 'None'}\n",
#         title="Current Progress", 
#         border_style="yellow"
#     ))


# Debug utility functions for pretty console output
def log_event(event_type, message, style="bold"):
    """Log an event with colored formatting based on its type"""
    colors = {
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "data": "magenta",
        "file": "cyan"
    }
    color = colors.get(event_type.lower(), "white")
    console.print(f"[bold {color}]{event_type.upper()}:[/] {message}", style=style)
    
def log_separator():
    """Print a visual separator in the console"""
    console.print("─" * 80, style="dim")


# Add performance tracking for response time
def log_performance_metrics(start_time, operation_name="Operation"):
    """Log performance metrics for an operation"""
    elapsed = time.time() - start_time
    console.print(f"[bold cyan]{operation_name} completed in:[/] {elapsed:.2f} seconds")
    
    if elapsed > 5:
        console.print("[yellow]Note: Operation took longer than expected[/]")
    
    # Display a mini chart
    blocks = min(20, int(elapsed * 2))
    bar = "█" * blocks
    console.print(f"[dim]Performance: [{'green' if elapsed < 3 else 'yellow' if elapsed < 10 else 'red'}]{bar}[/]")


# Run with: chainlit run chat.py -w --port 8080
if __name__ == "__main__":
    print_divider("STARTING CHAT APPLICATION")
    console.print("[bold green]This file is meant to be run with Chainlit:[/]")
    console.print("[cyan]chainlit run chat.py -w[/]")
    console.print(Panel("Debug Mode Enabled", title="Rich Console Logger", border_style="green"))
