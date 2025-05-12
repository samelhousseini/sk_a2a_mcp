import asyncio
import json
import uuid
import os
import logging
from typing import List, Dict, Any, Optional, AsyncIterable, Union, Callable, Tuple, cast

# Adjust path to import from the common module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'python')))

# Import helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from a2a_agents.a2a_utils.a2a_helper_functions import (
    create_client_message,
    create_task_params,
    parse_message_parts,
    extract_first_content_by_type,
    extract_all_text_parts,
    extract_all_data_parts,
    extract_and_save_files,
    extract_text_from_part,
    extract_data_from_part,
    extract_file_from_part,
    process_streaming_files,
    save_file_from_part
)

try:
    from common.client.client import A2AClient
    from common.client.card_resolver import A2ACardResolver
except ImportError:
    print("ERROR: Could not import A2AClient or A2ACardResolver. Ensuring fallback import...")
    # This is a fallback, ideally the path above should work if structure is as expected
    from samples.python.common.client.client import A2AClient
    from samples.python.common.client.card_resolver import A2ACardResolver

from common.types import (
    Message, TextPart, DataPart, Part, FilePart,
    TaskSendParams,
    AgentCard,
    SendTaskResponse, SendTaskStreamingResponse, Task,
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent, 
    JSONRPCError,
    A2AClientHTTPError, A2AClientJSONError
)

from a2a_agents.a2a_cards_resolver import A2ACardsResolver


class BaseA2AClient(A2AClient):
    """
    A base client class that inherits from A2AClient and supports both standard and SSE communication.
    This class keeps track of all parts, messages, and status updates exchanged.
    """

    def __init__(
        self,
        agent_card: Optional[AgentCard] = None,
        url: Optional[str] = None,
        timeout: float = 60.0, 
        output_dir: str = "output",
        debug: bool = False,
    ):
        """
        Initialize the BaseA2AClient.

        Args:
            agent_card: Optional AgentCard object with URL and other information
            url: URL of the agent server (if agent_card is not provided)
            timeout: Timeout for HTTP requests in seconds
            output_dir: Directory to store output files
            debug: Enable debug logging
        """
        # Call the parent constructor
        super().__init__(agent_card=agent_card, url=url, timeout=timeout)
        
        # Set up logging
        self.debug = debug
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("BaseA2AClient")
        
        # Storage for conversation history and tracking
        self.output_dir = output_dir
        self.messages_history: Dict[str, List[Message]] = {}  # task_id -> messages
        self.status_history: Dict[str, List[TaskStatusUpdateEvent]] = {}  # task_id -> status updates
        self.artifacts_history: Dict[str, List[TaskArtifactUpdateEvent]] = {}  # task_id -> artifacts
        self.file_paths: Dict[str, List[str]] = {}  # task_id -> saved file paths
        self.current_status: Dict[str, str] = {}  # task_id -> status
        
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    def _log(self, message: str) -> None:
        """Log a message if debug is enabled."""
        if self.debug:
            self.logger.debug(message)

    def _initialize_task_storage(self, task_id: str) -> None:
        """Initialize storage for a new task."""
        if task_id not in self.messages_history:
            self.messages_history[task_id] = []
        if task_id not in self.status_history:
            self.status_history[task_id] = []
        if task_id not in self.artifacts_history:
            self.artifacts_history[task_id] = []
        if task_id not in self.file_paths:
            self.file_paths[task_id] = []

    def _store_message(self, task_id: str, message: Message) -> None:
        """Store a message in the history."""
        self._initialize_task_storage(task_id)
        self.messages_history[task_id].append(message)
        
        # Save any file parts to disk
        self._save_file_parts(task_id, message)

    def _store_status_update(self, task_id: str, status_update: TaskStatusUpdateEvent) -> None:
        """Store a status update in the history."""
        self._initialize_task_storage(task_id)
        self.status_history[task_id].append(status_update)
        self.current_status[task_id] = status_update.status.state

    def _store_artifact_update(self, task_id: str, artifact_update: TaskArtifactUpdateEvent) -> None:
        """Store an artifact update in the history."""
        self._initialize_task_storage(task_id)
        self.artifacts_history[task_id].append(artifact_update)

    def _save_file_parts(self, task_id: str, message: Message) -> None:
        """Save any file parts in a message to disk."""
        for part in message.parts:
            if isinstance(part, FilePart):
                task_output_dir = os.path.join(self.output_dir, task_id)
                saved_path = save_file_from_part(part, task_output_dir)
                if saved_path:
                    self.file_paths[task_id].append(saved_path)
                    self._log(f"Saved file to {saved_path}")

    async def send_task_standard(self, task_params: TaskSendParams) -> Tuple[str, SendTaskResponse]:
        """
        Send a task using standard (non-streaming) communication.
        
        Args:
            task_params: Task parameters 
            
        Returns:
            Tuple containing:
                - task_id: The ID of the created task
                - response: The SendTaskResponse object
        """
        self._log(f"Sending standard task with parameters: {task_params}")
        # Call the parent method to send the task
        response = await self.send_task(task_params.model_dump())
        task_id = response.result.id
        
        # Initialize storage and store the client message
        self._initialize_task_storage(task_id)
        for message in response.result.history if response.result.history else []:
            self._store_message(task_id, message)
            
        self._log(f"Received task response for task_id: {task_id}")
        
        return task_id, response

    async def send_task_streaming(self, task_params: TaskSendParams) -> Tuple[str, AsyncIterable[SendTaskStreamingResponse]]:
        """
        Send a task using streaming (SSE) communication.
        
        Args:
            task_params: Task parameters
            
        Returns:
            Tuple containing:
                - task_id: The ID of the created task (or None if not yet received)
                - generator: An async generator that yields streaming responses
        """
        self._log(f"Sending streaming task with parameters: {task_params}")
        
        # Initialize a placeholder task_id that will be updated during processing
        outer_task_id = "pending"
          # Create a response processor that processes responses and yields them
        async def process_responses() -> AsyncIterable[SendTaskStreamingResponse]:
            nonlocal outer_task_id
            task_id = None          # Call the parent class's send_task_streaming method
            # We need to use the parent class method directly, not through super() in a nested function
            # This method returns an async generator directly - do not await it
            parent_stream = A2AClient.send_task_streaming(self, task_params.model_dump())
            
            # Process each response as it arrives
            async for response in parent_stream:
                # For the first response, get the task_id
                if response.result and hasattr(response.result, 'id'):
                    task_id = response.result.id
                    self._initialize_task_storage(task_id)
                    # Update the outer task_id that will be returned
                    outer_task_id = task_id
                
                # Process different types of streaming responses
                if response.result:
                    if hasattr(response.result, 'status'):  # Status update
                        if hasattr(response.result, 'model_dump'):
                            status_update = TaskStatusUpdateEvent(**response.result.model_dump())
                        else:
                            # Handle case where dict() might be used instead
                            status_update = response.result
                        self._store_status_update(task_id, status_update)
                    
                    elif hasattr(response.result, 'artifact'):  # Artifact update
                        if hasattr(response.result, 'model_dump'):
                            artifact_update = TaskArtifactUpdateEvent(**response.result.model_dump())
                        else:
                            # Handle case where dict() might be used instead
                            artifact_update = response.result
                        self._store_artifact_update(task_id, artifact_update)
                    
                    elif hasattr(response.result, 'history'):  # Message update for new API
                        for message in response.result.history:
                            self._store_message(task_id, message)
                
                yield response
        
        # Return the task_id (which will be updated during processing) and the response generator
        return outer_task_id, process_responses()

    def get_task_history(self, task_id: str) -> Dict[str, Any]:
        """
        Get the complete history for a task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            A dictionary containing all history for the task
        """
        return {
            "messages": self.messages_history.get(task_id, []),
            "status_updates": self.status_history.get(task_id, []),
            "artifacts": self.artifacts_history.get(task_id, []),
            "files": self.file_paths.get(task_id, []),
            "current_status": self.current_status.get(task_id)
        }

    def get_task_messages(self, task_id: str) -> List[Message]:
        """Get all messages for a task."""
        return self.messages_history.get(task_id, [])
    
    def get_latest_message(self, task_id: str) -> Optional[Message]:
        """Get the latest message for a task."""
        messages = self.messages_history.get(task_id, [])
        return messages[-1] if messages else None
    
    def get_task_files(self, task_id: str) -> List[str]:
        """Get all file paths saved from a task."""
        return self.file_paths.get(task_id, [])

    def get_current_status(self, task_id: str) -> Optional[str]:
        """Get the current status of a task."""
        return self.current_status.get(task_id)
    
    @staticmethod
    def create_client_message(
        text_content: Optional[Union[str, List[str]]] = None,
        data_content: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        file_paths: Optional[List[str]] = None,
        file_bytes_list: Optional[List[Tuple[bytes, Optional[str], Optional[str]]]] = None,
        file_uris: Optional[List[Tuple[str, Optional[str], Optional[str]]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Create a client message with multiple modalities (text, data, and files).
        
        Args:
            text_content: String or list of strings for text parts
            data_content: Dict or list of dicts for data parts
            file_paths: List of file paths to include as file parts
            file_bytes_list: List of (bytes, filename, mime_type) tuples for file parts
            file_uris: List of (uri, filename, mime_type) tuples for file parts
            metadata: Optional metadata to attach to all parts
            
        Returns:
            A Message object with all the specified parts
        """
        from a2a_agents.a2a_utils.a2a_helper_functions import (
            create_client_message as helper_create_client_message
        )
        
        # Use the helper function from a2a_helper_functions.py
        return helper_create_client_message(
            text_content=text_content,
            data_content=data_content,
            file_paths=file_paths,
            file_bytes_list=file_bytes_list,
            file_uris=file_uris,
            metadata=metadata
        )

    @staticmethod
    async def get_agent_cards_from_url(base_url: str) -> List[AgentCard]:
        """
        Get all agent cards from a server.
        
        Args:
            base_url: The base URL of the server
            
        Returns:
            List of agent cards
        """
        resolver = A2ACardsResolver(base_url)
        try:
            return resolver.get_agent_cards()
        except Exception as e:
            logging.error(f"Failed to get agent cards: {e}")
            # Try falling back to the standard resolver for a single card
            try:
                card_resolver = A2ACardResolver(base_url)
                card = card_resolver.get_agent_card()
                return [card]
            except Exception as e2:
                logging.error(f"Failed to get single agent card: {e2}")
                raise e2
