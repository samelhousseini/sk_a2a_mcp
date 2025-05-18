import requests
import asyncio
import uuid
import base64
import os.path
import json
from datetime import datetime
import sys
import copy
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Tuple, BinaryIO

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.traceback import install as install_rich_traceback
console = Console()
install_rich_traceback(show_locals=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'python')))

from common.types import (
    AgentCard, AgentSkill, AgentCapabilities, AgentAuthentication, AgentProvider,
    Task, TaskState, TaskStatus, Message, TextPart, DataPart, Part, FilePart, FileContent,
    Artifact, TaskSendParams, SendTaskRequest, SendTaskResponse, 
    SendTaskStreamingRequest, SendTaskStreamingResponse, 
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent,
    A2AClientHTTPError, A2AClientJSONError,
)
from common.server import A2AServer, InMemoryTaskManager


# Helper functions for creating message parts of different modalities
def create_text_part(text_content: str, metadata: Optional[Dict[str, Any]] = None) -> TextPart:
    """Creates a TextPart for sending text content"""
    return TextPart(type="text", text=text_content, metadata=metadata)

def create_data_part(data_content: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> DataPart:
    """Creates a DataPart for sending structured data content"""
    return DataPart(type="data", data=data_content, metadata=metadata)

def create_file_part_from_bytes(file_bytes: bytes, file_name: Optional[str] = None, 
                               mime_type: Optional[str] = None, 
                               metadata: Optional[Dict[str, Any]] = None) -> FilePart:
    """Creates a FilePart from raw bytes"""
    base64_content = base64.b64encode(file_bytes).decode('utf-8')
    file_content = FileContent(
        name=file_name,
        mimeType=mime_type,
        bytes=base64_content
    )
    return FilePart(type="file", file=file_content, metadata=metadata)

def create_file_part_from_path(file_path: str, mime_type: Optional[str] = None, 
                              metadata: Optional[Dict[str, Any]] = None) -> FilePart:
    """Creates a FilePart from a file path"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_name = os.path.basename(file_path)
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    return create_file_part_from_bytes(
        file_bytes=file_bytes,
        file_name=file_name,
        mime_type=mime_type,
        metadata=metadata
    )

def create_file_part_from_uri(uri: str, file_name: Optional[str] = None, 
                             mime_type: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> FilePart:
    """Creates a FilePart from a URI"""
    file_content = FileContent(
        name=file_name,
        mimeType=mime_type,
        uri=uri
    )
    return FilePart(type="file", file=file_content, metadata=metadata)

def create_message(role: str, parts: List[Part], metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Creates a Message with the specified role and parts"""
    return Message(role=role, parts=parts, metadata=metadata)

# Helper functions for parsing message parts
def extract_text_from_part(part: Part) -> Optional[str]:
    """Extracts text from a Part if it's a TextPart, otherwise returns None"""
    if isinstance(part, TextPart):
        return part.text
    return None

def extract_data_from_part(part: Part) -> Optional[Dict[str, Any]]:
    """Extracts data from a Part if it's a DataPart, otherwise returns None"""
    if isinstance(part, DataPart):
        return part.data
    return None

def extract_file_from_part(part: Part) -> Optional[Tuple[Union[bytes, str], Optional[str], Optional[str]]]:
    """
    Extracts file information from a Part if it's a FilePart, otherwise returns None.
    Returns a tuple of (file_content, file_name, mime_type) where file_content is either bytes or a URI.
    """
    if isinstance(part, FilePart):
        file_content = part.file
        if file_content.bytes:
            try:
                raw_bytes = base64.b64decode(file_content.bytes)
                return (raw_bytes, file_content.name, file_content.mimeType)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 file content: {e}")
        elif file_content.uri:
            return (file_content.uri, file_content.name, file_content.mimeType)
    return None

def save_file_from_part(part: Part, output_dir: str) -> Optional[str]:
    """
    Saves a file from a FilePart to the specified directory.
    Returns the path to the saved file if successful, None otherwise.
    """
    file_info = extract_file_from_part(part)
    if file_info:
        file_content, file_name, _ = file_info
        
        # Generate a filename if one is not provided
        if not file_name:
            file_name = f"file_{uuid.uuid4().hex}"
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        
        if isinstance(file_content, bytes):
            # Write binary content to file
            with open(output_path, 'wb') as f:
                f.write(file_content)
            return output_path
        elif isinstance(file_content, str):  # URI
            # For URI, we just return the URI string
            return file_content
    return None

# Helper functions for both standard and SSE communication
async def update_task_status(task_manager, task_id: str, state: TaskState, 
                            message_part: Optional[Part] = None,
                            text_message: Optional[str] = None) -> Task:
    """Updates a task's status for standard (non-streaming) communication"""
    # Create parts list from provided part or text_message
    parts_list: List[Part] = []
    if message_part:
        parts_list.append(message_part)
    elif text_message:
        parts_list.append(create_text_part(text_message))
    else:
        # Default empty message
        parts_list.append(create_text_part(""))
    
    agent_message = Message(role="agent", parts=parts_list)
    
    if task_id not in task_manager.tasks:
        raise ValueError(f"Task {task_id} not found")

    current_task = copy.deepcopy(task_manager.tasks[task_id])
    current_task.status.state = state
    current_task.status.message = agent_message
    current_task.status.timestamp = datetime.utcnow()
    if current_task.history is not None:
        current_task.history.append(agent_message)
    else:
        current_task.history = [agent_message]
    
    return current_task


async def add_task_artifact(task_manager, task_id: str, name: str, parts: Optional[List[Part]] = None,
                           text_content: Optional[str] = None, 
                           data_content: Optional[Dict[str, Any]] = None,
                           file_part: Optional[FilePart] = None) -> Task:
    """Adds an artifact to a task for standard (non-streaming) communication"""
    parts_list: List[Part] = parts or []
    
    # Add individual parts if provided
    if text_content: 
        parts_list.append(create_text_part(text_content))

    if data_content:
        parts_list.append(create_data_part(data_content))

    if file_part:
        parts_list.append(file_part)
    
    if not parts_list: 
        raise ValueError("Either parts, text_content, data_content, or file_part must be provided")
    
    artifact_obj = Artifact(name=name, parts=parts_list)
    
    if task_id not in task_manager.tasks:
        raise ValueError(f"Task {task_id} not found when adding artifact '{name}'")
        
    current_task = task_manager.tasks[task_id]
    if current_task.artifacts is None:
        current_task.artifacts = []
    current_task.artifacts.append(artifact_obj)
    
    return current_task


# Helper functions for SSE communication
async def send_status_update_event(task_manager, task_id: str, state: TaskState, 
                                  message_part: Optional[Part] = None,
                                  text_message: Optional[str] = None,
                                  data_content: Optional[Dict[str, Any]] = None,
                                  file_part: Optional[FilePart] = None,
                                  final: bool = False):
    """Sends a status update event for SSE streaming communication with support for all modalities"""
    # Create parts list from provided part or text_message
    parts_list: List[Part] = []
    if message_part:
        parts_list.append(message_part)
    elif text_message:
        parts_list.append(create_text_part(text_message))
    elif data_content:
        parts_list.append(create_data_part(data_content))
    elif file_part:
        parts_list.append(file_part)
    else:
        # Default empty message
        parts_list.append(create_text_part(""))
    
    agent_message = Message(role="agent", parts=parts_list)
    
    if task_id not in task_manager.tasks:
        error_status = TaskStatus(state=TaskState.FAILED, message=Message(role='agent',parts=[create_text_part('Task not found during processing')]))
        error_event = TaskStatusUpdateEvent(id=task_id, status=error_status, final=True)
        await task_manager.enqueue_events_for_sse(task_id, error_event)
        console.print(f"[bold red]send_status_update_event::Error:[/] Task {task_id} not found when sending status update event.\nText: {text_message}")
        return

    current_task = copy.deepcopy(task_manager.tasks[task_id])
    current_task.status.state = state
    current_task.status.message = agent_message
    current_task.status.timestamp = datetime.now()
    if current_task.history is not None:
        current_task.history.append(agent_message)
    else:
        current_task.history = [agent_message]
    
    status_event = TaskStatusUpdateEvent(id=task_id, status=current_task.status, final=final)

    message = f"[bold green]send_status_update_event::Status Update:[/] Task {task_id} updated to state {state.name} with message:\n {text_message or 'No message provided'}\nStatus: {current_task.status}\nFinal: {final}"
    console.print(Panel(message, title="Status Update Event Message", border_style="blue"))

    await task_manager.enqueue_events_for_sse(task_id, status_event)

async def add_task_artifact_event(task_manager, task_id: str, name: str, 
                                 parts: Optional[List[Part]] = None,
                                 text_content: Optional[str] = None, 
                                 data_content: Optional[Dict[str, Any]] = None,
                                 file_part: Optional[FilePart] = None,
                                 final_artifact_chunk: bool = True):
    """Adds an artifact event for SSE streaming communication with support for all modalities"""
    parts_list: List[Part] = parts or []
    
    # Add individual parts if provided
    if text_content: 
        parts_list.append(create_text_part(text_content))
    if data_content: 
        parts_list.append(create_data_part(data_content))
    if file_part:
        parts_list.append(file_part)
    
    if not parts_list: 
        # Nothing to send
        return
    
    artifact_obj = Artifact(name=name, parts=parts_list)
    
    if task_id not in task_manager.tasks:
        await send_status_update_event(
            task_manager, task_id, TaskState.WORKING, 
            text_message=f"Error: Task not found when adding artifact '{name}'.", 
            final=False
        )
        return
        
    current_task = task_manager.tasks[task_id]
    if current_task.artifacts is None:
        current_task.artifacts = []
    current_task.artifacts.append(artifact_obj)
    actual_added_artifact = current_task.artifacts[-1]
    actual_added_artifact.lastChunk = final_artifact_chunk
    artifact_event = TaskArtifactUpdateEvent(id=task_id, artifact=actual_added_artifact)
    await task_manager.enqueue_events_for_sse(task_id, artifact_event)


# Client-side helper functions for receiving and parsing messages
def parse_message_parts(message: Message) -> Dict[str, Any]:
    """
    Parse message parts and return them categorized by type.
    Returns a dictionary containing text_parts, data_parts, and file_parts.
    """
    result = {
        'text_parts': [],
        'data_parts': [],
        'file_parts': [],
    }
    
    for part in message.parts:
        if isinstance(part, TextPart):
            result['text_parts'].append(part)
        elif isinstance(part, DataPart):
            result['data_parts'].append(part)
        elif isinstance(part, FilePart):
            result['file_parts'].append(part)
    
    return result

def extract_first_content_by_type(message: Message, content_type: str) -> Any:
    """
    Extract the first content of a specific type from a message.
    content_type should be one of: 'text', 'data', 'file'
    Returns the content if found, None otherwise.
    """
    for part in message.parts:
        if content_type == 'text' and isinstance(part, TextPart):
            return part.text
        elif content_type == 'data' and isinstance(part, DataPart):
            return part.data
        elif content_type == 'file' and isinstance(part, FilePart):
            return extract_file_from_part(part)
    
    return None

def extract_all_file_parts(message: Message) -> List[Tuple[Union[bytes, str], Optional[str], Optional[str]]]:
    """
    Extract all file parts from a message.
    Returns a list of tuples: (file_content, file_name, mime_type)
    """
    result = []
    for part in message.parts:
        file_info = extract_file_from_part(part)
        if file_info:
            result.append(file_info)
    return result

def extract_all_text_parts(message: Message) -> List[str]:
    """Extract all text contents from a message."""
    result = []
    for part in message.parts:
        text = extract_text_from_part(part)
        if text:
            result.append(text)
    return result

def extract_all_data_parts(message: Message) -> List[Dict[str, Any]]:
    """Extract all data contents from a message."""
    result = []
    for part in message.parts:
        data = extract_data_from_part(part)
        if data:
            result.append(data)
    return result

# Combined helper function for creating client messages with mixed modalities
def create_client_message(
    text_content: Optional[Union[str, List[str]]] = None,
    data_content: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    file_paths: Optional[List[str]] = None,
    file_bytes_list: Optional[List[Tuple[bytes, Optional[str], Optional[str]]]] = None,
    file_uris: Optional[List[Tuple[str, Optional[str], Optional[str]]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """
    Create a client message with multiple modalities.
    
    Args:
        text_content: String or list of strings to include as text parts
        data_content: Dict or list of dicts to include as data parts
        file_paths: List of file paths to include as file parts
        file_bytes_list: List of tuples (bytes, name, mime_type) to include as file parts
        file_uris: List of tuples (uri, name, mime_type) to include as file parts
        metadata: Optional metadata for the message
    
    Returns:
        Message object with all specified parts
    """
    parts: List[Part] = []
    
    # Add text parts
    if text_content:
        if isinstance(text_content, str):
            parts.append(create_text_part(text_content))
        else:
            for text in text_content:
                parts.append(create_text_part(text))
    
    # Add data parts
    if data_content:
        if isinstance(data_content, dict):
            parts.append(create_data_part(data_content))
        else:
            for data in data_content:
                parts.append(create_data_part(data))
    
    # Add file parts from paths
    if file_paths:
        for file_path in file_paths:
            parts.append(create_file_part_from_path(file_path))
    
    # Add file parts from bytes
    if file_bytes_list:
        for file_bytes, name, mime_type in file_bytes_list:
            parts.append(create_file_part_from_bytes(file_bytes, name, mime_type))
    
    # Add file parts from URIs
    if file_uris:
        for uri, name, mime_type in file_uris:
            parts.append(create_file_part_from_uri(uri, name, mime_type))
    
    return Message(role="user", parts=parts, metadata=metadata)


# Helper functions for creating task parameters for client requests
def create_task_params(
    message: Message, 
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    history_length: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> TaskSendParams:
    """
    Create parameters for an agent task request.
    
    Args:
        message: The message to send
        task_id: Optional task ID (generated if not provided)
        session_id: Optional session ID
        history_length: Optional number of history messages to retrieve
        metadata: Optional metadata for the task
        
    Returns:
        TaskSendParams object for the request
    """
    params = {
        "id": task_id or uuid.uuid4().hex,
        "message": message
    }
    
    if session_id is not None:
        params["sessionId"] = session_id
    if history_length is not None:
        params["historyLength"] = history_length
    if metadata is not None:
        params["metadata"] = metadata
        
    return TaskSendParams(**params)


# Additional helper functions for client applications

async def upload_file_to_agent(
    a2a_client,  # A2AClient instance
    file_path: str,
    description: Optional[str] = None,
    mime_type: Optional[str] = None,
    additional_text: Optional[str] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Any:  # Returns response from the A2AClient.send_task method
    """
    Uploads a file to an agent using the A2A protocol.
    
    Args:
        a2a_client: An instance of A2AClient
        file_path: Path to the file to upload
        description: Optional text description to accompany the file
        mime_type: Optional MIME type for the file (will be guessed if not provided)
        additional_text: Optional text to include as a separate part
        task_id: Optional task ID (generated if not provided)
        session_id: Optional session ID
        metadata: Optional metadata for the task
        
    Returns:
        The response from the A2AClient.send_task method
    """
    # Create file part
    file_part = create_file_part_from_path(file_path, mime_type)
    
    # Create message with parts
    parts = []
    
    # Add text description if provided
    if description:
        parts.append(create_text_part(description))
    
    # Add additional text if provided
    if additional_text:
        parts.append(create_text_part(additional_text))
    
    # Add file part
    parts.append(file_part)
    
    # Create message and task parameters
    message = create_message(role="user", parts=parts)
    task_params = create_task_params(
        message=message,
        task_id=task_id,
        session_id=session_id,
        metadata=metadata
    )
    
    # Send the task
    return await a2a_client.send_task(payload=task_params.model_dump(exclude_none=True))


def extract_and_save_files(
    task: Task, 
    output_dir: str,
    status_only: bool = False
) -> List[str]:
    """
    Extract and save all files from a task's status message and/or artifacts.
    
    Args:
        task: The task containing file parts
        output_dir: Directory where files should be saved
        status_only: If True, only extract files from task status message
        
    Returns:
        List of paths to saved files
    """
    saved_files = []
    
    # Get files from status message
    if task.status and task.status.message:
        for part in task.status.message.parts:
            if isinstance(part, FilePart):
                file_path = save_file_from_part(part, output_dir)
                if file_path:
                    saved_files.append(file_path)
    
    # Get files from artifacts if requested
    if not status_only and task.artifacts:
        for artifact in task.artifacts:
            for part in artifact.parts:
                if isinstance(part, FilePart):
                    file_path = save_file_from_part(part, output_dir)
                    if file_path:
                        saved_files.append(file_path)
    
    return saved_files


async def upload_file_with_streaming_response(
    a2a_client,  # A2AClient instance
    file_path: str,
    description: Optional[str] = None,
    mime_type: Optional[str] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    process_event_callback: Optional[callable] = None
) -> AsyncGenerator[Any, None]:
    """
    Uploads a file to an agent using the A2A protocol with streaming response.
    
    Args:
        a2a_client: An instance of A2AClient
        file_path: Path to the file to upload
        description: Optional text description to accompany the file
        mime_type: Optional MIME type for the file (will be guessed if not provided)
        task_id: Optional task ID (generated if not provided)
        session_id: Optional session ID
        metadata: Optional metadata for the task
        process_event_callback: Optional callback function to process each event
        
    Yields:
        Events from the A2AClient.send_task_streaming method
    """
    # Create file part
    file_part = create_file_part_from_path(file_path, mime_type)
    
    # Create message with parts
    parts = []
    
    # Add text description if provided
    if description:
        parts.append(create_text_part(description))
    
    # Add file part
    parts.append(file_part)
    
    # Create message and task parameters
    message = create_message(role="user", parts=parts)
    task_params = create_task_params(
        message=message,
        task_id=task_id,
        session_id=session_id,
        metadata=metadata
    )
    
    # Send the task with streaming response
    async for event in a2a_client.send_task_streaming(payload=task_params.model_dump(exclude_none=True)):
        # Process the event with the callback if provided
        if process_event_callback:
            process_event_callback(event)
        
        yield event


def process_streaming_files(
    event,  # SendTaskStreamingResponse 
    output_dir: str,
    file_callback: Optional[callable] = None
) -> Optional[str]:
    """
    Process streaming response events and extract files when they appear.
    
    Args:
        event: A SendTaskStreamingResponse event
        output_dir: Directory where files should be saved
        file_callback: Optional callback function that takes (file_path, file_info)
        
    Returns:
        Path to a saved file if any was found in this event, None otherwise
    """
    saved_file = None
    
    if not event.result:
        return None
    
    # Process status update events with file parts
    if isinstance(event.result, TaskStatusUpdateEvent):
        status_event = event.result
        if status_event.status and status_event.status.message:
            for part in status_event.status.message.parts:
                if isinstance(part, FilePart):
                    saved_file = save_file_from_part(part, output_dir)
                    if saved_file and file_callback:
                        file_info = extract_file_from_part(part)
                        file_callback(saved_file, file_info)
                    return saved_file
    
    # Process artifact update events with file parts
    elif isinstance(event.result, TaskArtifactUpdateEvent):
        artifact_event = event.result
        if artifact_event.artifact and artifact_event.artifact.parts:
            for part in artifact_event.artifact.parts:
                if isinstance(part, FilePart):
                    saved_file = save_file_from_part(part, output_dir)
                    if saved_file and file_callback:
                        file_info = extract_file_from_part(part)
                        file_callback(saved_file, file_info)
                    return saved_file
    
    return saved_file


# Response processing helper functions

def print_part_contents(part: Part, label: str = "Part", indent: str = "      "):
    """
    Print the contents of a message part based on its type.
    
    Args:
        part: The Part object to process
        label: Label to prepend to the output
        indent: String to use for indentation
    """
    text_content = extract_text_from_part(part)
    data_content = extract_data_from_part(part)
    file_content = extract_file_from_part(part)
    
    if text_content:
        print(f"{indent}{label} (Text): {text_content}")
    elif data_content:
        print(f"{indent}{label} (Data): {json.dumps(data_content, indent=2)}")
    elif file_content:
        file_data, file_name, mime_type = file_content
        if isinstance(file_data, bytes):
            size = len(file_data)
            print(f"{indent}{label} (File): {file_name or 'unnamed'}, Type: {mime_type or 'unknown'}, Size: {size} bytes")
        else:  # URI
            print(f"{indent}{label} (File URI): {file_data}, Name: {file_name or 'unnamed'}, Type: {mime_type or 'unknown'}")
    else:
        print(f"{indent}{label} (Unknown type)")


def process_standard_response(client, task_id: str, response: SendTaskResponse, print_output: bool = True, 
                             indent: str = "  ") -> Dict[str, Any]:
    """
    Process a standard (non-streaming) response from an A2A agent.
    
    Args:
        client: The BaseA2AClient instance that sent the request
        task_id: The ID of the task
        response: The SendTaskResponse object
        print_output: Whether to print processed information to console
        indent: String to use for indentation
        
    Returns:
        Dict containing processed information about the response
    """
    result = {
        "id": response.id,
        "task_id": task_id,
        "status": None,
        "messages": [],
        "error": None
    }
    
    if print_output:
        print(f"\nReceived Response:")
        print(f"{indent}JSON-RPC ID: {response.id}")
        print(f"{indent}Task ID: {task_id}")
    
    if response.result:
        # Get status
        status = client.get_current_status(task_id)
        result["status"] = status
        
        if print_output and status:
            print(f"{indent}Task Status: {status}")
        
        # Get messages
        messages = client.get_task_messages(task_id)
        agent_messages = [msg for msg in messages if msg.role == "agent"]
        result["messages"] = agent_messages
        
        if agent_messages and print_output:
            latest_message = agent_messages[-1]
            
            # Display text parts
            for part in latest_message.parts:
                text_content = extract_text_from_part(part)
                if text_content:
                    print(f"{indent}Text Message: {text_content}")
            
            # Display data parts
            for part in latest_message.parts:
                data_content = extract_data_from_part(part)
                if data_content:
                    print(f"{indent}Data Message: {json.dumps(data_content, indent=2)}")
        
        # Check for saved files
        file_paths = client.get_task_files(task_id)
        result["files"] = file_paths
        
        if file_paths and print_output:
            print(f"{indent}Files saved ({len(file_paths)}):")
            for i, path in enumerate(file_paths):
                print(f"{indent}  File {i+1}: {path}")
        
        # Get task history summary
        history = client.get_task_history(task_id)
        result["history"] = history
        
        if print_output:
            print(f"\n{indent}Message count: {len(history['messages'])}")
            print(f"{indent}Status updates: {len(history['status_updates'])}")
            print(f"{indent}Artifacts: {len(history['artifacts'])}")
    
    elif response.error:
        error_detail = response.error
        result["error"] = {
            "code": error_detail.code,
            "message": error_detail.message,
            "data": error_detail.data
        }
        
        if print_output:
            print(f"{indent}Error from Agent:")
            print(f"{indent}  Code: {error_detail.code}")
            print(f"{indent}  Message: {error_detail.message}")
            if error_detail.data:
                print(f"{indent}  Data: {json.dumps(error_detail.data, indent=2)}")
    
    return result


def process_streaming_response(response, print_output: bool = True, indent: str = "  ") -> Dict[str, Any]:
    """
    Process a single streaming response from an A2A agent.
    
    Args:
        response: A single SendTaskStreamingResponse object
        print_output: Whether to print processed information to console
        indent: String to use for indentation
        
    Returns:
        Dict containing processed information about the response
    """
    result = {
        "id": response.id,
        "type": None,
        "content": None,
        "is_final": False,
        "error": None
    }
    
    if print_output:
        print(f"\nReceived Streaming Response:")
        print(f"{indent}JSON-RPC ID: {response.id}")
    
    if response.result:
        event_result = response.result
        
        if hasattr(event_result, 'status'):
            # Status update event
            status = event_result.status
            result["type"] = "status"
            result["content"] = {
                "state": status.state,
                "message": status.message if hasattr(status, 'message') else None
            }
            result["is_final"] = event_result.final
            
            if print_output:
                print(f"{indent}Event Type: Status Update")
                print(f"{indent}  State: {status.state}")
                
                # Display message parts if available
                if hasattr(status, 'message') and status.message and status.message.parts:
                    print(f"{indent}  Message parts: {len(status.message.parts)}")
                    for i, part in enumerate(status.message.parts):
                        print_part_contents(part, f"Message part {i+1}", indent + "  ")
                        
                print(f"{indent}  Is Final: {event_result.final}")
                if event_result.final:
                    print(f"--- Final status update received. Agent processing complete. ---")
        
        elif hasattr(event_result, 'artifact'):
            # Artifact event
            artifact = event_result.artifact
            result["type"] = "artifact"
            result["content"] = {
                "name": artifact.name,
                "parts": artifact.parts,
                "lastChunk": artifact.lastChunk
            }
            result["is_final"] = artifact.lastChunk
            
            if print_output:
                print(f"{indent}Event Type: Artifact Update")
                print(f"{indent}  Artifact Name: {artifact.name}")
                
                # Display artifact parts
                if artifact.parts:
                    print(f"{indent}  Artifact parts: {len(artifact.parts)}")
                    for i, part in enumerate(artifact.parts):
                        print_part_contents(part, f"Artifact part {i+1}", indent + "  ")
                
                print(f"{indent}  Is Final Chunk: {artifact.lastChunk}")
        
        elif hasattr(event_result, 'delta'):
            # Delta event (for streaming message content)
            delta = event_result.delta
            result["type"] = "delta"
            result["content"] = {
                "role": delta.role,
                "parts": delta.parts
            }
            
            if print_output:
                print(f"{indent}Event Type: Message Delta")
                print(f"{indent}  Role: {delta.role}")
                
                for part in delta.parts:
                    print_part_contents(part, "Delta part", indent + "  ")
        
        else:
            result["type"] = "unknown"
            if print_output:
                print(f"{indent}Unknown event structure in result")
    
    elif response.error:
        error_detail = response.error
        result["error"] = {
            "code": error_detail.code,
            "message": error_detail.message,
            "data": error_detail.data if hasattr(error_detail, 'data') else None
        }
        
        if print_output:
            print(f"{indent}Error Event from Agent:")
            print(f"{indent}  Code: {error_detail.code}")
            print(f"{indent}  Message: {error_detail.message}")
            if hasattr(error_detail, 'data') and error_detail.data:
                print(f"{indent}  Data: {json.dumps(error_detail.data, indent=2)}")
    
    return result


def print_task_summary(client, task_id: str, indent: str = "  ") -> Dict[str, Any]:
    """
    Print a summary of the task status and return the information.
    
    Args:
        client: The BaseA2AClient instance
        task_id: The task ID to summarize
        indent: String to use for indentation
        
    Returns:
        Dict containing summary information
    """
    if not task_id or task_id == "pending":
        print(f"{indent}No valid task ID available for summary")
        return {}
    
    print(f"\n--- Task Summary ---")
    history = client.get_task_history(task_id)
    
    print(f"{indent}Message count: {len(history['messages'])}")
    print(f"{indent}Status updates: {len(history['status_updates'])}")
    print(f"{indent}Artifacts: {len(history['artifacts'])}")
    print(f"{indent}Current status: {history['current_status']}")
    
    # Check for saved files
    file_paths = client.get_task_files(task_id)
    if file_paths:
        print(f"\n{indent}Files saved ({len(file_paths)}):")
        for i, path in enumerate(file_paths):
            print(f"{indent}  {i+1}: {path}")
    
    return {
        "history": history,
        "file_paths": file_paths
    }


def handle_client_error(e: Exception, error_type: str = "A2A Client", indent: str = "  ") -> Dict[str, Any]:
    """
    Handle and print client errors in a consistent way.
    
    Args:
        e: The exception to handle
        error_type: Type of error for display
        indent: String to use for indentation
        
    Returns:
        Dict with error information
    """
    result = {
        "type": error_type,
        "message": str(e)
    }
    
    print(f"\n--- {error_type} Error --- ")
    
    if isinstance(e, A2AClientHTTPError):
        result["status_code"] = e.status_code
        print(f"{indent}Status Code: {e.status_code}")
        print(f"{indent}Message: {e.message}")
    elif isinstance(e, A2AClientJSONError):
        print(f"{indent}Message: {e.message}")
    else:
        print(f"{indent}Error details: {e}")
        import traceback
        traceback.print_exc()
        result["traceback"] = traceback.format_exc()
    
    return result


# Additional helper functions for agent message processing

def extract_user_query(agent_query: Dict[str, Any], log_prefix: str = None, logger_instance=None) -> Tuple[str, bool]:
    """
    Extract user query text from agent_query.
    
    Args:
        agent_query: The dictionary containing query data
        log_prefix: Optional prefix for log messages (e.g., "[AgentName - taskId]")
        logger_instance: Optional logger to use for warnings
        
    Returns:
        Tuple of (extracted_text, is_empty_flag)
    """
    query_text_parts = agent_query.get("text_contents", [])
    user_query_text = ""
    
    if query_text_parts:
        if isinstance(query_text_parts[0], str):
            user_query_text = query_text_parts[0].strip()
        else:
            # Try to join all parts as strings
            user_query_text = " ".join([str(part) for part in query_text_parts]).strip()
    
    is_empty = not bool(user_query_text)
    
    if is_empty and logger_instance and log_prefix:
        logger_instance.warning(f"{log_prefix} No user query text found.")
        
    return user_query_text, is_empty

def create_agent_response(
    text_content: Optional[str] = None, 
    data_content: Optional[Dict[str, Any]] = None,
    file_content: Optional[Union[
        str,  # File path
        Tuple[bytes, Optional[str], Optional[str]],  # (file_bytes, file_name, mime_type)
        Tuple[str, Optional[str], Optional[str]],  # (file_uri, file_name, mime_type)
        FilePart  # Existing FilePart object
    ]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """
    Create a standard agent response message with optional text, data, and file parts.
    
    Args:
        text_content: Optional text content for the response
        data_content: Optional dictionary for structured data in the response
        file_content: Optional file content which can be:
            - A file path string
            - A tuple of (file_bytes, file_name, mime_type)
            - A tuple of (file_uri, file_name, mime_type)
            - An existing FilePart object
        metadata: Optional metadata for the message
        
    Returns:
        Message object with appropriate parts
    """
    parts = []
    
    # Add text part if provided
    if text_content:
        parts.append(TextPart(type="text", text=text_content))
    
    # Add data part if provided
    if data_content:
        parts.append(DataPart(type="data", data=data_content))
    
    # Add file part if provided
    if file_content:
        if isinstance(file_content, str):
            # Assume it's a file path
            file_part = create_file_part_from_path(file_content)
            parts.append(file_part)
        elif isinstance(file_content, tuple):
            if isinstance(file_content[0], bytes):
                # (file_bytes, file_name, mime_type)
                file_bytes, file_name, mime_type = file_content
                file_part = create_file_part_from_bytes(file_bytes, file_name, mime_type)
                parts.append(file_part)
            elif isinstance(file_content[0], str):
                # (file_uri, file_name, mime_type)
                file_uri, file_name, mime_type = file_content
                file_part = create_file_part_from_uri(file_uri, file_name, mime_type)
                parts.append(file_part)
        elif isinstance(file_content, FilePart):
            # Directly use the provided FilePart
            parts.append(file_content)
    
    # Ensure we have at least one part if nothing was provided
    if not parts:
        parts.append(TextPart(type="text", text=""))
    
    return Message(role="agent", parts=parts, metadata=metadata)

def handle_empty_query(agent_name: str, task_id: str, 
                      message: str = "I didn't receive a question. Please provide a question.", 
                      logger_instance=None) -> Message:
    """
    Create a standard response for empty query situations.
    
    Args:
        agent_name: Name of the agent for logging
        task_id: Task ID for logging
        message: Custom message to return
        logger_instance: Optional logger for warnings
        
    Returns:
        Message object with appropriate parts
    """
    if logger_instance:
        logger_instance.warning(f"[{agent_name} - {task_id}] No user query text found.")
    return create_agent_response(message)

def log_agent_response(agent_name: str, task_id: str, response_text: str, 
                      data_content: Optional[Dict[str, Any]] = None,
                      logger_instance=None) -> None:
    """
    Log an agent's response with standardized formatting.
    
    Args:
        agent_name: Name of the agent
        task_id: Current task ID
        response_text: Text part of the response
        data_content: Optional data part for logging
        logger_instance: Logger to use
    """
    if not logger_instance:
        return
        
    if data_content:
        data_str = str(data_content).replace("'", '"')
        logger_instance.info(f"[{agent_name} - {task_id}] Formulated response: {response_text}, Data: {data_str}")
    else:
        logger_instance.info(f"[{agent_name} - {task_id}] Formulated response: {response_text}")

def extract_message_content(agent_query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all types of content from agent_query into a structured dictionary.
    
    Args:
        agent_query: The dictionary containing query data
        
    Returns:
        Dictionary containing extracted text, data, and file contents
    """
    result = {
        'text': "",
        'data': {},
        'files': []
    }
    
    # Extract text content
    query_text_parts = agent_query.get("text_contents", [])
    if query_text_parts:
        if isinstance(query_text_parts[0], str):
            result['text'] = query_text_parts[0].strip()
        else:
            result['text'] = " ".join([str(part) for part in query_text_parts]).strip()
    
    # Extract data content
    query_data_parts = agent_query.get("data_contents", [])
    if query_data_parts and isinstance(query_data_parts[0], dict):
        result['data'] = query_data_parts[0]
    
    # Extract file content
    query_file_parts = agent_query.get("file_contents", [])
    if query_file_parts:
        result['files'] = query_file_parts
    
    return result