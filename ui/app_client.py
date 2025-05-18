#!/usr/bin/env python
"""
App Client for Customer Support Chat

This client handles communication with the Orchestrator Agent and processes
various message types (text, data, files) for the UI.
"""

import os
import sys
import uuid
import logging
import base64
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncGenerator
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.traceback import install as install_rich_traceback
console = Console()
install_rich_traceback(show_locals=True)

# Add parent directory to path so we can import a2a_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from a2a_agents.base_a2a_client import BaseA2AClient
    from a2a_agents.a2a_utils.a2a_helper_functions import (
        create_task_params, create_client_message, extract_message_content,
        extract_user_query, create_agent_response, log_agent_response, 
        process_streaming_response, extract_text_from_part, extract_data_from_part, 
        extract_file_from_part, save_file_from_part, create_text_part,
        create_data_part, create_file_part_from_bytes
    )
    from common.types import (
        Message, TextPart, DataPart, FilePart, Part, TaskSendParams,
        Task, TaskState, TaskStatus, TaskStatusUpdateEvent, TaskArtifactUpdateEvent,
        Artifact
    )
except ImportError as e:
    logging.error(f"Import error: {e}. Make sure the repository structure is correct.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default environment variables
ORCHESTRATOR_AGENT_URL = os.environ.get("ORCHESTRATOR_AGENT_URL", "http://localhost:8000")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ChatClient:
    """A class to handle communication with the Orchestrator Agent"""
    
    def __init__(self, url: str = ORCHESTRATOR_AGENT_URL, output_dir: str = OUTPUT_DIR):
        """Initialize the client"""
        self.url = url
        self.output_dir = output_dir
        self.client = BaseA2AClient(
            url=self.url,
            output_dir=self.output_dir,
            debug=True  # Enable debug logging
        )
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        logger.info(f"Chat client initialized. Target URL: {self.url}")
        
    async def chat(self, user_query: str, data_content: Optional[Dict[str, Any]] = None) -> Task:
        """
        Send a user query to the orchestrator agent and return the task response
        
        Args:
            user_query: The user's message
            data_content: Optional structured data to include with the message
        
        Returns:
            Task object containing all response content (text, data, files, artifacts, etc.)
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        logger.info(f"\n--- Sending Query to Orchestrator (Task ID: {task_id}) ---")
        logger.info(f"User Query: {user_query}")
        
        # Create client message with text and optional data parts
        message = create_client_message(
            text_content=user_query,
            data_content=data_content
        )
        
        # Create task parameters
        task_params = create_task_params(message, task_id=task_id, session_id=self.session_id)
        
        try:
            # Send task and get response
            returned_task_id, response = await self.client.send_task_standard(task_params)
            
            # Get the complete task with all information
            task = self.client.get_task(returned_task_id)
            
            # Log the response
            logger.info(f"Received response for Task ID: {returned_task_id}")
            latest_message = self.client.get_latest_message(returned_task_id)
            
            if latest_message and latest_message.parts:
                parts_summary = []
                for part in latest_message.parts:
                    if isinstance(part, TextPart):
                        parts_summary.append(f"Text ({len(part.text)} chars)")
                    elif isinstance(part, DataPart):
                        parts_summary.append(f"Data ({len(str(part.data))} chars)")
                    elif isinstance(part, FilePart):
                        parts_summary.append(f"File {part.file.name or 'unnamed'}")
                
                logger.info(f"Response parts: {', '.join(parts_summary)}")
            
            return task
                
        except Exception as e:
            logger.error(f"Error sending task to orchestrator: {e}", exc_info=True)
            
            # Create an error task to represent the failure
            error_parts = [create_text_part(f"I'm sorry, an error occurred: {str(e)}")]
            error_message = Message(role="agent", parts=error_parts)
            error_status = TaskStatus(state=TaskState.FAILED, message=error_message)
            
            # Construct a minimal task with error information
            error_task = Task(
                id=task_id,
                sessionId=self.session_id,
                status=error_status,
                artifacts=None,
                history=[message, error_message]  # Include the user's message and error response
            )
            
            return error_task
            
    async def chat_streaming(self, user_query: str, data_content: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent], None]:
        """
        Send a user query to the orchestrator agent and stream structured responses
        
        Args:
            user_query: The user's message
            data_content: Optional structured data to include with the message
        
        Yields:
            TaskStatusUpdateEvent or TaskArtifactUpdateEvent objects as chunks arrive
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        logger.info(f"\n--- Sending Streaming Query to Orchestrator (Task ID: {task_id}) ---")
        logger.info(f"User Query: {user_query}")
        
        # Create client message with text and optional data parts
        message = create_client_message(
            text_content=user_query,
            data_content=data_content
        )
        
        # Create task parameters
        task_params = create_task_params(message, task_id=task_id, session_id=self.session_id)
        
        try:
            # Send task and get streaming response
            returned_task_id, response_stream = await self.client.send_task_streaming(task_params)
            
            # Process the streaming response
            async for sse_response in response_stream:
                # Pass through the raw event directly
                yield sse_response.result
                
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}", exc_info=True)
            
            # Create an error status update to represent the failure
            error_parts = [create_text_part(f"I'm sorry, an error occurred: {str(e)}")]
            error_message = Message(role="agent", parts=error_parts)
            error_status = TaskStatus(state=TaskState.FAILED, message=error_message)
            
            # Yield the error status update
            yield TaskStatusUpdateEvent(
                id=task_id,
                status=error_status,
                final=True
            )

    async def extract_structured_data(self, user_query: str) -> Dict[str, Any]:
        """
        Extract structured data from a user query
        
        Args:
            user_query: The user's message
            
        Returns:
            Dictionary of extracted structured data
        """
        # Implementation to be added
        pass
