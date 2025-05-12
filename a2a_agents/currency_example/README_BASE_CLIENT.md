# Base A2A Client

This implementation provides a base client class (`BaseA2AClient`) for Agent-to-Agent (A2A) communication that supports both standard and Server-Sent Events (SSE) streaming communication.

## Requirements

- Python 3.8+
- httpx
- httpx-sse

## Installation

The base client is part of the A2A communication system. To use it:

1. Make sure you have the required dependencies:
```bash
pip install httpx httpx-sse
```

2. The client works with the existing A2A framework, so ensure you have the common types and helpers available in your project structure.

## Features

- Inherits from the original `A2AClient` class
- Supports both standard and SSE (streaming) communication modes
- Uses helper functions from `a2a_helper_functions.py`
- Tracks and stores all parts, messages, and status updates exchanged
- Automatically saves file attachments to disk
- Provides methods to retrieve message history and current status
- Works with both single and multiple agent card servers using `A2ACardsResolver`

## Files

- `base_a2a_client.py` - The main client implementation
- `base_client_example.py` - An example showing how to use the client with both communication modes

## How to Use

### Initialize the Client

```python
from base_a2a_client import BaseA2AClient

# Option 1: Using an agent card
agent_card = resolver.get_agent_card()
client = BaseA2AClient(agent_card=agent_card)

# Option 2: Using a URL directly
client = BaseA2AClient(url="http://localhost:8001")

# Additional configuration options
client = BaseA2AClient(
    agent_card=agent_card,
    timeout=120.0,  # HTTP timeout in seconds
    output_dir="client_output",  # Directory to save file attachments
    debug=True  # Enable debug logging
)
```

### Standard Communication

```python
from a2a_utils.a2a_helper_functions import create_task_params

# Create a message
message = BaseA2AClient.create_client_message(text_content="Hello agent")

# Create task parameters
task_params = create_task_params(message, session_id="my-session-id")

# Send the task (standard mode)
task_id, response = await client.send_task_standard(task_params)

# Get the response message
messages = client.get_task_messages(task_id)
latest_message = client.get_latest_message(task_id)
```

### Multi-Modal Communication

```python
# Create a multi-modal message with text, data, and files
message = BaseA2AClient.create_client_message(
    text_content=["Main message", "Additional text"],  # Can be a string or list of strings
    data_content={                                     # Can be a dict or list of dicts
        "example_data": {
            "values": [1, 2, 3, 4, 5],
            "metadata": {"source": "User input"}
        }
    },
    file_paths=["path/to/file.txt", "path/to/image.jpg"],  # List of file paths
    # Optional: provide raw file bytes with name and mime type
    file_bytes_list=[(bytes_data, "filename.dat", "application/octet-stream")],
    # Optional: provide file URIs
    file_uris=[("https://example.com/file.pdf", "remote_file.pdf", "application/pdf")]
)
```

### Streaming Communication

```python
# Create task parameters
message = BaseA2AClient.create_client_message(text_content="Generate step by step...")
task_params = create_task_params(message, session_id="my-session-id")

# Send the task (streaming mode)
task_id, response_stream = await client.send_task_streaming(task_params)

# Process streaming responses
async for response in response_stream:
    # The client automatically stores these updates
    # But you can process them here as well
    if hasattr(response.result, 'status'):
        status = response.result.status
        print(f"Status Update: {status.state}")
    elif hasattr(response.result, 'delta'):
        delta = response.result.delta
        # Process delta parts
        if delta.role == "agent":
            for part in delta.parts:
                if hasattr(part, 'text') and part.text:
                    print(f"Agent: {part.text}")
```

### Getting Task History and Status

```python
# Get all history for a task
history = client.get_task_history(task_id)

# Get just the messages
messages = client.get_task_messages(task_id)

# Get the current status
status = client.get_current_status(task_id)

# Get saved file paths
file_paths = client.get_task_files(task_id)
```

## Multiple Agent Support

The client works with the `A2ACardsResolver` to support retrieving multiple agent cards from a server:

```python
# Get all agent cards from a server
cards = await BaseA2AClient.get_agent_cards_from_url(base_url)

# Create a client for each card
for card in cards:
    client = BaseA2AClient(agent_card=card)
    # Use the client...
```

## File Handling

The client automatically handles extracting and saving files from agent responses:

```python
# Files are automatically saved to the output directory
file_paths = client.get_task_files(task_id)

# You can also process files directly
latest_message = client.get_latest_message(task_id)
if latest_message:
    from a2a_utils.a2a_helper_functions import extract_file_from_part
    
    for part in latest_message.parts:
        file_info = extract_file_from_part(part)
        if file_info:
            content, name, mime_type = file_info
            # Process file content
```

## Task History and Tracking

The client maintains a complete history of all messages, status updates, and artifacts:

```python
# Get complete history for a task
history = client.get_task_history(task_id)
print(f"Messages: {len(history['messages'])}")
print(f"Status updates: {len(history['status_updates'])}")
print(f"Artifacts: {len(history['artifacts'])}")
print(f"Files: {history['files']}")
print(f"Current status: {history['current_status']}")

# Get just the latest message
latest = client.get_latest_message(task_id)
```

## Running the Example

```bash
python base_client_example.py
```

This example demonstrates:
1. Retrieving agent cards from a server
2. Sending a standard request and processing the response 
3. Sending a streaming request and processing the updates
4. Creating and sending multi-modal messages (text, data, and files)
5. Accessing saved files and message history
