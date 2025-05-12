# A2A Python Master Reference

This document consolidates the core Python code for the A2A (Agent-to-Agent) communication protocol implementation, including common types, client, server, utilities, and a sample CLI host. It's intended as a reference for LLMs generating A2A-related code.

## Table of Contents

1.  [Common Types (`common/types.py`)](#common-types-commontypespy)
2.  [Common Utilities (`common/utils/`)](#common-utilities-commonutils)
    *   [In-Memory Cache (`in_memory_cache.py`)](#in-memory-cache-in_memory_cachepy)
    *   [Push Notification Auth (`push_notification_auth.py`)](#push-notification-auth-push_notification_authpy)
3.  [Common Client (`common/client/`)](#common-client-commonclient)
    *   [Card Resolver (`card_resolver.py`)](#card-resolver-card_resolverpy)
    *   [A2A Client (`client.py`)](#a2a-client-clientpy)
    *   [Client Init (`__init__.py`)](#client-init-__init__py)
4.  [Common Server (`common/server/`)](#common-server-commonserver)
    *   [Server Utilities (`utils.py`)](#server-utilities-utilspy)
    *   [Task Manager (`task_manager.py`)](#task-manager-task_managerpy)
    *   [A2A Server (`server.py`)](#a2a-server-serverpy)
    *   [Server Init (`__init__.py`)](#server-init-__init__py)
5.  [Sample CLI Host (`hosts/cli/`)](#sample-cli-host-hostscli)
    *   [Push Notification Listener (`push_notification_listener.py`)](#push-notification-listener-push_notification_listenerpy)
    *   [CLI Main (`__main__.py`)](#cli-main-__main__py)

---

## 1. Common Types (`common/types.py`)

Defines the Pydantic models used for data structures throughout the A2A protocol, including messages, tasks, artifacts, events, and error types.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\types.py
from typing import Union, Any
from pydantic import BaseModel, Field, TypeAdapter
from typing import Literal, List, Annotated, Optional
from datetime import datetime
from pydantic import model_validator, ConfigDict, field_serializer
from uuid import uuid4
from enum import Enum
from typing_extensions import Self


class TaskState(str, Enum):
    """Enumeration of possible task states."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    UNKNOWN = "unknown"


class TextPart(BaseModel):
    """Represents a text part of a message."""
    type: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] | None = None


class FileContent(BaseModel):
    """Represents the content of a file, either as bytes or a URI."""
    name: str | None = None
    mimeType: str | None = None
    bytes: str | None = None  # Base64 encoded bytes
    uri: str | None = None

    @model_validator(mode="after")
    def check_content(self) -> Self:
        """Ensures either 'bytes' or 'uri' is provided, but not both."""
        if not (self.bytes or self.uri):
            raise ValueError("Either 'bytes' or 'uri' must be present in the file data")
        if self.bytes and self.uri:
            raise ValueError(
                "Only one of 'bytes' or 'uri' can be present in the file data"
            )
        return self


class FilePart(BaseModel):
    """Represents a file part of a message."""
    type: Literal["file"] = "file"
    file: FileContent
    metadata: dict[str, Any] | None = None


class DataPart(BaseModel):
    """Represents a structured data part of a message."""
    type: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


# Union type for different message parts
Part = Annotated[Union[TextPart, FilePart, DataPart], Field(discriminator="type")]


class Message(BaseModel):
    """Represents a message exchanged between user and agent."""
    role: Literal["user", "agent"]
    parts: List[Part]
    metadata: dict[str, Any] | None = None


class TaskStatus(BaseModel):
    """Represents the status of a task at a point in time."""
    state: TaskState
    message: Message | None = None  # Optional message associated with the status update
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_serializer("timestamp")
    def serialize_dt(self, dt: datetime, _info):
        """Serializes datetime to ISO format string."""
        return dt.isoformat()


class Artifact(BaseModel):
    """Represents an artifact produced during a task."""
    name: str | None = None
    description: str | None = None
    parts: List[Part]
    metadata: dict[str, Any] | None = None
    index: int = 0  # For chunking large artifacts
    append: bool | None = None # Indicates if this chunk appends to the previous one
    lastChunk: bool | None = None # Indicates if this is the last chunk


class Task(BaseModel):
    """Represents a complete task, including its status, history, and artifacts."""
    id: str
    sessionId: str | None = None
    status: TaskStatus
    artifacts: List[Artifact] | None = None
    history: List[Message] | None = None # Conversation history for the task
    metadata: dict[str, Any] | None = None


class TaskStatusUpdateEvent(BaseModel):
    """Event sent (e.g., via SSE or push notification) when a task's status changes."""
    id: str # Task ID
    status: TaskStatus
    final: bool = False # Indicates if this is the final status update for the task
    metadata: dict[str, Any] | None = None


class TaskArtifactUpdateEvent(BaseModel):
    """Event sent when a new artifact (or chunk) is available for a task."""
    id: str # Task ID
    artifact: Artifact
    metadata: dict[str, Any] | None = None


class AuthenticationInfo(BaseModel):
    """Describes authentication requirements or details."""
    model_config = ConfigDict(extra="allow") # Allows extra fields if needed

    schemes: List[str] # e.g., ["bearer", "apiKey"]
    credentials: str | None = None # Actual credential (e.g., token, key) - use with care


class PushNotificationConfig(BaseModel):
    """Configuration for sending push notifications."""
    url: str # The endpoint URL to send notifications to
    token: str | None = None # Deprecated? Consider using AuthenticationInfo
    authentication: AuthenticationInfo | None = None # Authentication details for the push endpoint


class TaskIdParams(BaseModel):
    """Parameters containing just a task ID."""
    id: str
    metadata: dict[str, Any] | None = None


class TaskQueryParams(TaskIdParams):
    """Parameters for querying a task, including optional history length."""
    historyLength: int | None = None # Number of history messages to retrieve


class TaskSendParams(BaseModel):
    """Parameters for sending a new message/initiating a task."""
    id: str # Task ID (client-generated)
    sessionId: str = Field(default_factory=lambda: uuid4().hex) # Session ID
    message: Message # The message to send
    acceptedOutputModes: Optional[List[str]] = None # e.g., ["text", "image/png"]
    pushNotification: PushNotificationConfig | None = None # Optional push notification config
    historyLength: int | None = None # How much history to return in the initial response
    metadata: dict[str, Any] | None = None


class TaskPushNotificationConfig(BaseModel):
    """Parameters for setting or getting push notification config for a task."""
    id: str
    pushNotificationConfig: PushNotificationConfig


## RPC Messages

Defines the structure for JSON-RPC 2.0 requests and responses used in A2A.

class JSONRPCMessage(BaseModel):
    """Base model for JSON-RPC messages."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None = Field(default_factory=lambda: uuid4().hex) # Request ID


class JSONRPCRequest(JSONRPCMessage):
    """Represents a JSON-RPC request."""
    method: str # The method to be invoked (e.g., "tasks/send")
    params: dict[str, Any] | None = None # Parameters for the method


class JSONRPCError(BaseModel):
    """Represents a JSON-RPC error object."""
    code: int # Error code (standard or custom)
    message: str # Error message
    data: Any | None = None # Additional error data


class JSONRPCResponse(JSONRPCMessage):
    """Represents a JSON-RPC response."""
    result: Any | None = None # Result of the method call (if successful)
    error: JSONRPCError | None = None # Error object (if failed)


# Specific Request/Response types for A2A methods

class SendTaskRequest(JSONRPCRequest):
    method: Literal["tasks/send"] = "tasks/send"
    params: TaskSendParams


class SendTaskResponse(JSONRPCResponse):
    result: Task | None = None # Returns the initial task state


class SendTaskStreamingRequest(JSONRPCRequest):
    method: Literal["tasks/sendSubscribe"] = "tasks/sendSubscribe"
    params: TaskSendParams


class SendTaskStreamingResponse(JSONRPCResponse):
    # Result can be a status update or an artifact update during streaming
    result: TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None = None


class GetTaskRequest(JSONRPCRequest):
    method: Literal["tasks/get"] = "tasks/get"
    params: TaskQueryParams


class GetTaskResponse(JSONRPCResponse):
    result: Task | None = None # Returns the current task state


class CancelTaskRequest(JSONRPCRequest):
    method: Literal["tasks/cancel",] = "tasks/cancel" # Note the comma for single-element tuple
    params: TaskIdParams


class CancelTaskResponse(JSONRPCResponse):
    result: Task | None = None # Returns the task state after cancellation attempt


class SetTaskPushNotificationRequest(JSONRPCRequest):
    method: Literal["tasks/pushNotification/set",] = "tasks/pushNotification/set"
    params: TaskPushNotificationConfig


class SetTaskPushNotificationResponse(JSONRPCResponse):
    result: TaskPushNotificationConfig | None = None


class GetTaskPushNotificationRequest(JSONRPCRequest):
    method: Literal["tasks/pushNotification/get",] = "tasks/pushNotification/get"
    params: TaskIdParams


class GetTaskPushNotificationResponse(JSONRPCResponse):
    result: TaskPushNotificationConfig | None = None


class TaskResubscriptionRequest(JSONRPCRequest):
    method: Literal["tasks/resubscribe",] = "tasks/resubscribe"
    params: TaskIdParams


# Type adapter to parse incoming JSON-RPC requests based on the 'method' field
A2ARequest = TypeAdapter(
    Annotated[
        Union[
            SendTaskRequest,
            GetTaskRequest,
            CancelTaskRequest,
            SetTaskPushNotificationRequest,
            GetTaskPushNotificationRequest,
            TaskResubscriptionRequest,
            SendTaskStreamingRequest,
        ],
        Field(discriminator="method"),
    ]
)

## Error types

Standard and custom JSON-RPC error types for A2A.

class JSONParseError(JSONRPCError):
    code: int = -32700
    message: str = "Invalid JSON payload"
    data: Any | None = None


class InvalidRequestError(JSONRPCError):
    code: int = -32600
    message: str = "Request payload validation error"
    data: Any | None = None


class MethodNotFoundError(JSONRPCError):
    code: int = -32601
    message: str = "Method not found"
    data: None = None


class InvalidParamsError(JSONRPCError):
    code: int = -32602
    message: str = "Invalid parameters"
    data: Any | None = None


class InternalError(JSONRPCError):
    code: int = -32603
    message: str = "Internal error"
    data: Any | None = None


# A2A Specific Errors (using custom code range -32000 to -32099)

class TaskNotFoundError(JSONRPCError):
    code: int = -32001
    message: str = "Task not found"
    data: None = None


class TaskNotCancelableError(JSONRPCError):
    code: int = -32002
    message: str = "Task cannot be canceled"
    data: None = None


class PushNotificationNotSupportedError(JSONRPCError):
    code: int = -32003
    message: str = "Push Notification is not supported"
    data: None = None


class UnsupportedOperationError(JSONRPCError):
    code: int = -32004
    message: str = "This operation is not supported"
    data: None = None


class ContentTypeNotSupportedError(JSONRPCError):
    code: int = -32005
    message: str = "Incompatible content types"
    data: None = None


## Agent Card Models

Models defining the structure of an Agent Card (`/.well-known/agent.json`).

class AgentProvider(BaseModel):
    """Information about the agent provider."""
    organization: str
    url: str | None = None


class AgentCapabilities(BaseModel):
    """Flags indicating the agent's capabilities."""
    streaming: bool = False # Supports tasks/sendSubscribe
    pushNotifications: bool = False # Supports push notifications
    stateTransitionHistory: bool = False # Includes detailed state history


class AgentAuthentication(BaseModel):
    """Authentication requirements for the agent itself (if any)."""
    schemes: List[str]
    credentials: str | None = None


class AgentSkill(BaseModel):
    """Describes a specific skill or capability of the agent."""
    id: str
    name: str
    description: str | None = None
    tags: List[str] | None = None
    examples: List[str] | None = None # Example prompts or use cases
    inputModes: List[str] | None = None # Supported input MIME types/modes
    outputModes: List[str] | None = None # Supported output MIME types/modes


class AgentCard(BaseModel):
    """The main Agent Card model."""
    name: str
    description: str | None = None
    url: str # The base URL endpoint for the agent's A2A API
    provider: AgentProvider | None = None
    version: str
    documentationUrl: str | None = None
    capabilities: AgentCapabilities
    authentication: AgentAuthentication | None = None
    defaultInputModes: List[str] = ["text"]
    defaultOutputModes: List[str] = ["text"]
    skills: List[AgentSkill]


## Client Exception Types

Custom exceptions for the A2A client.

class A2AClientError(Exception):
    """Base exception for A2A client errors."""
    pass


class A2AClientHTTPError(A2AClientError):
    """Exception for HTTP errors encountered by the client."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP Error {status_code}: {message}")


class A2AClientJSONError(A2AClientError):
    """Exception for JSON parsing errors encountered by the client."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"JSON Error: {message}")


class MissingAPIKeyError(Exception):
    """Exception for missing API key (potentially used by specific agents)."""
    pass

```

---

## 2. Common Utilities (`common/utils/`)

Helper classes and functions used by both client and server components.

### In-Memory Cache (`in_memory_cache.py`)

A simple thread-safe singleton in-memory cache, potentially used for storing task data or session information on the server side.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\utils\in_memory_cache.py
"""In Memory Cache utility."""

import threading
import time
from typing import Any, Dict, Optional


class InMemoryCache:
    """A thread-safe Singleton class to manage cache data.

    Ensures only one instance of the cache exists across the application.
    """

    _instance: Optional["InMemoryCache"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls):
        """Override __new__ to control instance creation (Singleton pattern).

        Uses a lock to ensure thread safety during the first instantiation.

        Returns:
            The singleton instance of InMemoryCache.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the cache storage.

        Uses a flag (_initialized) to ensure this logic runs only on the very first
        creation of the singleton instance.
        """
        # This check prevents re-initialization on subsequent calls to __init__
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    # print("Initializing SessionCache storage")
                    self._cache_data: Dict[str, Dict[str, Any]] = {} # Stores the actual cache items
                    self._ttl: Dict[str, float] = {} # Stores expiration timestamps
                    self._data_lock: threading.Lock = threading.Lock() # Lock for accessing cache data
                    self._initialized = True

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a key-value pair with an optional time-to-live (TTL).

        Args:
            key: The key for the data.
            value: The data to store.
            ttl: Time to live in seconds. If None, data will not expire.
        """
        with self._data_lock:
            self._cache_data[key] = value

            if ttl is not None:
                self._ttl[key] = time.time() + ttl
            else:
                # Remove any existing TTL if ttl is set to None
                if key in self._ttl:
                    del self._ttl[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value associated with a key, checking for expiration.

        Args:
            key: The key for the data.
            default: The value to return if the key is not found or expired.

        Returns:
            The cached value, or the default value if not found or expired.
        """
        with self._data_lock:
            # Check if the key has expired
            if key in self._ttl and time.time() > self._ttl[key]:
                # Expired: delete from cache and TTL dict
                del self._cache_data[key]
                del self._ttl[key]
                return default
            # Return the value if found and not expired, or the default
            return self._cache_data.get(key, default)

    def delete(self, key: str) -> bool:
        """Delete a specific key-value pair from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        with self._data_lock:
            if key in self._cache_data:
                del self._cache_data[key]
                # Also remove from TTL dict if present
                if key in self._ttl:
                    del self._ttl[key]
                return True
            return False # Key not found

    def clear(self) -> bool:
        """Remove all data from the cache."""
        with self._data_lock:
            self._cache_data.clear()
            self._ttl.clear()
            return True
        # This return False seems unreachable given the lock, but kept for consistency
        # return False

```

### Push Notification Auth (`push_notification_auth.py`)

Handles the authentication and verification logic for push notifications using JWT and JWK sets. Includes classes for both the sender (agent) and receiver (client/host).

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\utils\push_notification_auth.py
from jwcrypto import jwk # For JWK generation
import uuid
from starlette.responses import JSONResponse
from starlette.requests import Request
from typing import Any

import jwt # PyJWT library for JWT operations
import time
import json
import hashlib # For SHA256 hashing
import httpx # For making HTTP requests (e.g., to verify URLs, fetch JWKS)
import logging

from jwt import PyJWK, PyJWKClient # Specific classes from PyJWT

logger = logging.getLogger(__name__)
AUTH_HEADER_PREFIX = 'Bearer ' # Standard prefix for Bearer tokens

class PushNotificationAuth:
    """Base class with common utility for calculating request body hash."""
    def _calculate_request_body_sha256(self, data: dict[str, Any]) -> str:
        """Calculates the SHA256 hash of a JSON request body.

        Ensures consistent serialization (no extra spaces, sorted keys implicitly by json.dumps)
        before hashing. This is crucial for the signature to match between sender and receiver.
        """
        body_str = json.dumps(
            data,
            ensure_ascii=False, # Keep non-ASCII characters as is
            allow_nan=False, # Standard JSON doesn't allow NaN/Infinity
            indent=None, # No indentation
            separators=(",", ":"), # Compact separators
        )
        return hashlib.sha256(body_str.encode('utf-8')).hexdigest() # Hash the UTF-8 encoded string

class PushNotificationSenderAuth(PushNotificationAuth):
    """Handles authentication logic for the agent sending push notifications."""
    def __init__(self):
        self.public_keys: List[dict] = [] # Stores public keys in JWK format (as dicts)
        self.private_key_jwk: PyJWK = None # Stores the current private key as a PyJWK object

    @staticmethod
    async def verify_push_notification_url(url: str) -> bool:
        """Verifies the client's push notification endpoint URL.

        Sends a GET request with a validation token and expects the token back.
        This confirms the client controls the endpoint before sending sensitive data.
        """
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                validation_token = str(uuid.uuid4())
                response = await client.get(
                    url,
                    params={"validationToken": validation_token} # Send token as query param
                )
                response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
                is_verified = response.text == validation_token # Check if response body matches token

                logger.info(f"Verified push-notification URL: {url} => {is_verified}")
                return is_verified
            except Exception as e:
                # Log errors during verification (network issues, client errors)
                logger.warning(f"Error during sending push-notification URL verification for {url}: {e}")

        return False # Verification failed

    def generate_jwk(self):
        """Generates a new RSA key pair (JWK format) for signing JWTs.

        Adds the public key to the list and stores the private key.
        Uses 'sig' (signature) use case.
        """
        key = jwk.JWK.generate(kty='RSA', size=2048, kid=str(uuid.uuid4()), use="sig")
        self.public_keys.append(key.export_public(as_dict=True)) # Store public key as dict
        self.private_key_jwk = PyJWK.from_json(key.export_private()) # Store private key as PyJWK object
        logger.info(f"Generated new JWK with kid: {self.private_key_jwk.key_id}")

    def handle_jwks_endpoint(self, _request: Request) -> JSONResponse:
        """Provides a JWKS (JSON Web Key Set) endpoint.

        Allows clients (receivers) to fetch the agent's public keys to verify JWT signatures.
        Typically exposed at `/.well-known/jwks.json`.
        """
        return JSONResponse({
            "keys": self.public_keys # Return the list of public keys
        })

    def _generate_jwt(self, data: dict[str, Any]) -> str:
        """Generates a JWT for a push notification payload.

        The JWT includes:
        - 'iat' (issued at): Timestamp to prevent replay attacks.
        - 'request_body_sha256': SHA256 hash of the notification payload to ensure integrity.
        The JWT is signed using the agent's private key.
        The 'kid' (Key ID) header is included to help the receiver find the correct public key.
        """
        if not self.private_key_jwk:
             self.generate_jwk() # Generate a key if none exists

        iat = int(time.time()) # Issued at timestamp

        payload_to_encode = {
            "iat": iat,
            "request_body_sha256": self._calculate_request_body_sha256(data)
        }

        headers = {"kid": self.private_key_jwk.key_id} # Include Key ID in header

        return jwt.encode(
            payload_to_encode,
            key=self.private_key_jwk.key, # Use the underlying key object from PyJWK
            algorithm="RS256", # Use RSA SHA-256 algorithm
            headers=headers
        )

    async def send_push_notification(self, url: str, data: dict[str, Any]):
        """Sends a push notification to the specified URL.

        Generates a JWT, includes it in the Authorization header, and POSTs the data.
        """
        jwt_token = self._generate_jwt(data)
        headers = {'Authorization': f'{AUTH_HEADER_PREFIX}{jwt_token}'} # Format as Bearer token
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(
                    url,
                    json=data, # Send original data as JSON body
                    headers=headers
                )
                response.raise_for_status() # Check for HTTP errors
                logger.info(f"Push-notification sent successfully to URL: {url}")
            except Exception as e:
                # Log errors during sending
                logger.warning(f"Error during sending push-notification to URL {url}: {e}")

class PushNotificationReceiverAuth(PushNotificationAuth):
    """Handles authentication logic for the client/host receiving push notifications."""
    def __init__(self):
        # self.public_keys_jwks = [] # Not needed directly, PyJWKClient handles fetching/caching
        self.jwks_client: PyJWKClient | None = None # Client to fetch JWKS from the agent

    async def load_jwks(self, jwks_url: str):
        """Initializes the JWKS client with the agent's JWKS endpoint URL."""
        self.jwks_client = PyJWKClient(jwks_url)
        logger.info(f"Initialized JWKS client for URL: {jwks_url}")
        # Optionally pre-fetch keys here if desired, but PyJWKClient fetches on demand
        # try:
        #     await self.jwks_client.fetch_data() # httpx needs to run in async context
        # except Exception as e:
        #     logger.error(f"Failed to fetch initial JWKS data from {jwks_url}: {e}")


    async def verify_push_notification(self, request: Request) -> bool:
        """Verifies an incoming push notification request.

        1. Extracts the JWT from the Authorization header.
        2. Fetches the correct public key from the agent's JWKS endpoint using the 'kid' from the JWT header.
        3. Decodes and verifies the JWT signature, 'iat' claim, and 'request_body_sha256' claim.
        4. Compares the hash in the JWT with the hash of the actual request body.
        5. Checks if the 'iat' timestamp is within an acceptable window (e.g., 5 minutes).
        """
        if not self.jwks_client:
            logger.error("JWKS client not loaded. Call load_jwks first.")
            return False

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith(AUTH_HEADER_PREFIX):
            logger.warning("Invalid or missing Authorization header in push notification.")
            return False

        token = auth_header[len(AUTH_HEADER_PREFIX):] # Extract token part

        try:
            # Get the signing key from the JWKS endpoint based on the token's header
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)

            # Decode the JWT, verifying signature and required claims
            decoded_token = jwt.decode(
                token,
                signing_key.key, # Use the underlying key object
                options={
                    "require": ["iat", "request_body_sha256"] # Ensure these claims are present
                },
                algorithms=["RS256"], # Specify allowed algorithm(s)
            )

            # Verify the request body integrity
            actual_body_sha256 = self._calculate_request_body_sha256(await request.json())
            if actual_body_sha256 != decoded_token["request_body_sha256"]:
                logger.warning("Push notification body hash mismatch.")
                raise ValueError("Invalid request body hash") # Body integrity check failed

            # Verify the timestamp to prevent replay attacks
            max_age_seconds = 60 * 5 # 5 minutes tolerance
            if time.time() - decoded_token["iat"] > max_age_seconds:
                logger.warning(f"Push notification token expired (issued at {decoded_token['iat']}).")
                raise ValueError("Token is expired") # Replay attack prevention

            logger.info("Push notification verified successfully.")
            return True

        except jwt.exceptions.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token in push notification: {e}")
        except httpx.RequestError as e:
             logger.error(f"Failed to fetch JWKS data: {e}")
        except Exception as e:
            # Catch other potential errors (e.g., network issues fetching JWKS, JSON errors)
            logger.error(f"Error verifying push notification: {e}")

        return False # Verification failed

```

---

## 3. Common Client (`common/client/`)

Code related to the A2A client-side implementation, used by host applications to interact with agents.

### Card Resolver (`card_resolver.py`)

A simple utility to fetch and parse an agent's Agent Card from its well-known endpoint.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\client\card_resolver.py
import httpx # For making HTTP requests
from common.types import (
    AgentCard, # Pydantic model for the Agent Card
    A2AClientJSONError, # Custom exception for JSON errors
    A2AClientHTTPError, # Custom exception for HTTP errors (though not explicitly raised here, good practice)
)
import json # For handling potential JSONDecodeError

class A2ACardResolver:
    """Fetches and parses an Agent Card from a base URL."""
    def __init__(self, base_url: str, agent_card_path: str = "/.well-known/agent.json"):
        """Initializes the resolver with the agent's base URL and the card path.

        Args:
            base_url: The base URL of the agent (e.g., "http://localhost:10000").
            agent_card_path: The path to the agent card JSON file (default: "/.well-known/agent.json").
        """
        self.base_url = base_url.rstrip("/") # Remove trailing slash if present
        self.agent_card_path = agent_card_path.lstrip("/") # Remove leading slash if present
        self.full_url = f"{self.base_url}/{self.agent_card_path}"

    def get_agent_card(self) -> AgentCard:
        """Synchronously fetches and parses the Agent Card.

        Raises:
            A2AClientHTTPError: If the HTTP request fails (e.g., 404, 500).
            A2AClientJSONError: If the response body is not valid JSON or doesn't match the AgentCard model.

        Returns:
            An AgentCard object parsed from the JSON response.
        """
        try:
            # Use httpx.Client for potential connection pooling if used frequently
            with httpx.Client() as client:
                response = client.get(self.full_url)
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
                # Attempt to parse the JSON response into an AgentCard model
                return AgentCard(**response.json())
        except httpx.HTTPStatusError as e:
            # Wrap HTTP errors in our custom exception
            raise A2AClientHTTPError(e.response.status_code, f"Failed to fetch agent card from {self.full_url}: {e}") from e
        except json.JSONDecodeError as e:
            # Wrap JSON decoding errors
            raise A2AClientJSONError(f"Failed to decode JSON agent card from {self.full_url}: {e}") from e
        except Exception as e: # Catch other potential errors (e.g., network issues)
             raise A2AClientError(f"An unexpected error occurred while fetching agent card from {self.full_url}: {e}") from e

    async def get_agent_card_async(self) -> AgentCard:
        """Asynchronously fetches and parses the Agent Card.

        Raises:
            A2AClientHTTPError: If the HTTP request fails (e.g., 404, 500).
            A2AClientJSONError: If the response body is not valid JSON or doesn't match the AgentCard model.

        Returns:
            An AgentCard object parsed from the JSON response.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.full_url)
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
                # Attempt to parse the JSON response into an AgentCard model
                return AgentCard(**response.json())
        except httpx.HTTPStatusError as e:
            # Wrap HTTP errors in our custom exception
            raise A2AClientHTTPError(e.response.status_code, f"Failed to fetch agent card from {self.full_url}: {e}") from e
        except json.JSONDecodeError as e:
            # Wrap JSON decoding errors
            raise A2AClientJSONError(f"Failed to decode JSON agent card from {self.full_url}: {e}") from e
        except Exception as e: # Catch other potential errors (e.g., network issues)
             raise A2AClientError(f"An unexpected error occurred while fetching agent card from {self.full_url}: {e}") from e

```

### A2A Client (`client.py`)

The main client class for interacting with an A2A agent. It provides methods for sending tasks (blocking and streaming), getting task status, canceling tasks, and managing push notifications.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\client\client.py
import httpx # Async HTTP client
from httpx_sse import connect_sse # For handling Server-Sent Events (SSE)
from typing import Any, AsyncIterable
from common.types import (
    AgentCard,
    GetTaskRequest,
    SendTaskRequest,
    SendTaskResponse,
    JSONRPCRequest, # Base JSON-RPC request model
    JSONRPCResponse, # Base JSON-RPC response model
    GetTaskResponse,
    CancelTaskResponse,
    CancelTaskRequest,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    A2AClientHTTPError, # Custom HTTP error
    A2AClientJSONError, # Custom JSON error
    A2AClientError, # Base client error
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
)
import json
import logging

logger = logging.getLogger(__name__)

class A2AClient:
    """Client for interacting with an A2A agent API."""
    def __init__(self, agent_card: AgentCard = None, url: str = None, timeout: float = 30.0):
        """Initializes the client with either an AgentCard or a direct URL.

        Args:
            agent_card: An optional AgentCard object. If provided, its `url` field is used.
            url: An optional direct URL to the agent's A2A endpoint. Used if `agent_card` is None.
            timeout: Default timeout in seconds for non-streaming requests.

        Raises:
            ValueError: If neither `agent_card` nor `url` is provided.
        """
        if agent_card:
            self.url = agent_card.url.rstrip("/")
        elif url:
            self.url = url.rstrip("/")
        else:
            raise ValueError("Must provide either agent_card or url")
        self.timeout = timeout
        logger.info(f"A2AClient initialized for agent at: {self.url}")

    async def _send_request(self, request: JSONRPCRequest) -> dict[str, Any]:
        """Internal helper to send a JSON-RPC request and handle responses/errors."""
        async with httpx.AsyncClient() as client:
            try:
                logger.debug(f"Sending request {request.method} ({request.id}) to {self.url}")
                response = await client.post(
                    self.url,
                    json=request.model_dump(exclude_none=True), # Send Pydantic model as JSON
                    timeout=self.timeout
                )
                response.raise_for_status() # Raise exception for 4xx/5xx status codes
                json_response = response.json()
                logger.debug(f"Received response for request {request.id}: {json_response}")

                # Basic check for JSON-RPC response structure
                if 'jsonrpc' not in json_response or json_response['jsonrpc'] != '2.0':
                     raise A2AClientJSONError("Invalid JSON-RPC version in response")
                if 'id' not in json_response or json_response['id'] != request.id:
                     raise A2AClientJSONError("Mismatched JSON-RPC ID in response")
                if 'result' not in json_response and 'error' not in json_response:
                     raise A2AClientJSONError("Invalid JSON-RPC response: missing 'result' or 'error'")

                return json_response
            except httpx.HTTPStatusError as e:
                # Log and wrap HTTP errors
                logger.error(f"HTTP error {e.response.status_code} for {request.method} ({request.id}): {e.response.text}")
                raise A2AClientHTTPError(e.response.status_code, str(e)) from e
            except json.JSONDecodeError as e:
                # Log and wrap JSON decoding errors
                logger.error(f"JSON decode error for {request.method} ({request.id}): {e}")
                raise A2AClientJSONError(str(e)) from e
            except httpx.RequestError as e: # Catch network errors (connection, timeout, etc.)
                 logger.error(f"Network error for {request.method} ({request.id}): {e}")
                 raise A2AClientError(f"Network error contacting agent: {e}") from e
            except Exception as e: # Catch unexpected errors
                 logger.exception(f"Unexpected error during request {request.method} ({request.id})")
                 raise A2AClientError(f"An unexpected error occurred: {e}") from e

    async def send_task(self, payload: dict[str, Any]) -> SendTaskResponse:
        """Sends a task request (non-streaming).

        Args:
            payload: Dictionary conforming to TaskSendParams.

        Returns:
            A SendTaskResponse object containing the initial task state or an error.
        """
        try:
            request = SendTaskRequest(params=payload)
            response_dict = await self._send_request(request)
            return SendTaskResponse(**response_dict)
        except Exception as e: # Catch potential validation errors during request creation
            logger.error(f"Error creating SendTaskRequest: {e}")
            # Return a response object with an internal error representation
            return SendTaskResponse(id=payload.get("id"), error={"code": -32603, "message": f"Client-side request creation error: {e}"})


    async def send_task_streaming(
        self, payload: dict[str, Any]
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        """Sends a task request and subscribes to streaming updates (SSE).

        Args:
            payload: Dictionary conforming to TaskSendParams.

        Yields:
            SendTaskStreamingResponse objects containing task updates or errors.

        Raises:
            A2AClientError: If the connection fails or invalid data is received.
        """
        try:
            request = SendTaskStreamingRequest(params=payload)
            # Use a longer timeout or None for streaming connections
            async with httpx.AsyncClient(timeout=None) as client:
                try:
                    # `connect_sse` handles the SSE connection and event parsing
                    with connect_sse(
                        client, "POST", self.url, json=request.model_dump(exclude_none=True)
                    ) as event_source:
                        logger.info(f"SSE connection established for task {payload.get('id')}")
                        for sse in event_source.iter_sse():
                            if sse.event == "error": # Handle potential error events from server
                                logger.error(f"SSE error event received: {sse.data}")
                                # You might want to parse sse.data if it's a JSONRPCError structure
                                yield SendTaskStreamingResponse(id=request.id, error={"code": -32000, "message": f"SSE error: {sse.data}"})
                                break # Stop iteration on server error event
                            elif sse.event == "message": # Default event type
                                try:
                                    logger.debug(f"SSE data received: {sse.data}")
                                    # Parse the data as a JSON-RPC response containing the update
                                    response_data = json.loads(sse.data)
                                    yield SendTaskStreamingResponse(**response_data)
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to decode SSE data: {e} - Data: '{sse.data}'")
                                    yield SendTaskStreamingResponse(id=request.id, error={"code": -32700, "message": f"Invalid JSON in SSE data: {e}"})
                                    # Decide whether to break or continue on decode error
                                    # break
                                except Exception as e: # Catch validation errors etc.
                                     logger.error(f"Error processing SSE data: {e}")
                                     yield SendTaskStreamingResponse(id=request.id, error={"code": -32603, "message": f"Error processing SSE data: {e}"})
                                     # break
                            else:
                                logger.warning(f"Received unknown SSE event type: {sse.event}")

                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error during SSE connection: {e.response.status_code} - {e.response.text}")
                    raise A2AClientHTTPError(e.response.status_code, f"SSE connection failed: {e}") from e
                except httpx.RequestError as e:
                    logger.error(f"Network error during SSE connection: {e}")
                    raise A2AClientError(f"SSE connection network error: {e}") from e
                except Exception as e:
                    logger.exception("Unexpected error during SSE streaming")
                    raise A2AClientError(f"Unexpected SSE error: {e}") from e
        except Exception as e: # Catch potential validation errors during request creation
            logger.error(f"Error creating SendTaskStreamingRequest: {e}")
            # For async iterables, raising is often better than yielding an error and stopping.
            # Or, yield one error response and then stop.
            yield SendTaskStreamingResponse(id=payload.get("id"), error={"code": -32603, "message": f"Client-side request creation error: {e}"})
            # raise A2AClientError(f"Client-side request creation error: {e}") from e


    async def get_task(self, payload: dict[str, Any]) -> GetTaskResponse:
        """Gets the current status and details of a task.

        Args:
            payload: Dictionary conforming to TaskQueryParams.

        Returns:
            A GetTaskResponse object containing the task details or an error.
        """
        try:
            request = GetTaskRequest(params=payload)
            response_dict = await self._send_request(request)
            return GetTaskResponse(**response_dict)
        except Exception as e:
            logger.error(f"Error creating GetTaskRequest: {e}")
            return GetTaskResponse(id=payload.get("id"), error={"code": -32603, "message": f"Client-side request creation error: {e}"})

    async def cancel_task(self, payload: dict[str, Any]) -> CancelTaskResponse:
        """Requests cancellation of a running task.

        Args:
            payload: Dictionary conforming to TaskIdParams.

        Returns:
            A CancelTaskResponse object indicating the result or an error.
        """
        try:
            request = CancelTaskRequest(params=payload)
            response_dict = await self._send_request(request)
            return CancelTaskResponse(**response_dict)
        except Exception as e:
            logger.error(f"Error creating CancelTaskRequest: {e}")
            return CancelTaskResponse(id=payload.get("id"), error={"code": -32603, "message": f"Client-side request creation error: {e}"})

    async def set_task_push_notification(
        self, payload: dict[str, Any]
    ) -> SetTaskPushNotificationResponse:
        """Sets or updates the push notification configuration for a task.

        Args:
            payload: Dictionary conforming to TaskPushNotificationConfig.

        Returns:
            A SetTaskPushNotificationResponse object confirming the config or an error.
        """
        try:
            request = SetTaskPushNotificationRequest(params=payload)
            response_dict = await self._send_request(request)
            return SetTaskPushNotificationResponse(**response_dict)
        except Exception as e:
            logger.error(f"Error creating SetTaskPushNotificationRequest: {e}")
            return SetTaskPushNotificationResponse(id=payload.get("id"), error={"code": -32603, "message": f"Client-side request creation error: {e}"})

    async def get_task_push_notification(
        self, payload: dict[str, Any]
    ) -> GetTaskPushNotificationResponse:
        """Gets the current push notification configuration for a task.

        Args:
            payload: Dictionary conforming to TaskIdParams.

        Returns:
            A GetTaskPushNotificationResponse object containing the config or an error.
        """
        try:
            request = GetTaskPushNotificationRequest(params=payload)
            response_dict = await self._send_request(request)
            return GetTaskPushNotificationResponse(**response_dict)
        except Exception as e:
            logger.error(f"Error creating GetTaskPushNotificationRequest: {e}")
            return GetTaskPushNotificationResponse(id=payload.get("id"), error={"code": -32603, "message": f"Client-side request creation error: {e}"})

    # Note: Resubscribe functionality might need adjustments based on server implementation
    # async def resubscribe_to_task(self, payload: dict[str, Any]) -> AsyncIterable[SendTaskStreamingResponse]:
    #     """Resubscribes to an existing task's event stream."""
    #     # Implementation would be similar to send_task_streaming but using TaskResubscriptionRequest
    #     # and potentially handling initial state differently.
    #     request = TaskResubscriptionRequest(params=payload)
    #     # ... rest similar to send_task_streaming ...
    #     pass

```

### Client Init (`__init__.py`)

Makes the core client classes easily importable from the `common.client` package.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\client\__init__.py
from .client import A2AClient
from .card_resolver import A2ACardResolver

# Expose the main classes for easier import
__all__ = ["A2AClient", "A2ACardResolver"]
```

---

## 4. Common Server (`common/server/`)

Code related to the A2A server-side implementation, used by agents to expose the A2A API.

### Server Utilities (`utils.py`)

Helper functions specifically for the server side, like checking modality compatibility and creating standard error responses.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\server\utils.py
from common.types import (
    JSONRPCResponse,
    ContentTypeNotSupportedError, # Specific error for modality mismatch
    UnsupportedOperationError, # Specific error for unimplemented features
)
from typing import List

def are_modalities_compatible(
    server_output_modes: List[str] | None, client_output_modes: List[str] | None
) -> bool:
    """Checks if the client's accepted output modes overlap with the server's capabilities.

    Args:
        server_output_modes: List of MIME types the server *can* produce for this task/skill.
        client_output_modes: List of MIME types the client *will accept* (from `acceptedOutputModes`).

    Returns:
        True if compatible, False otherwise. Compatibility rules:
        - Always True if client_output_modes is None or empty (client accepts anything/default).
        - Always True if server_output_modes is None or empty (server implies default, e.g., text).
          (Consider if a server *must* declare its modes - spec might clarify).
        - True if there's at least one common mode in both lists.
        - False otherwise.
    """
    # If client doesn't specify, assume compatibility (accepts default, likely 'text')
    if client_output_modes is None or not client_output_modes:
        return True

    # If server doesn't specify its modes for this skill/task, assume default ('text') compatibility
    # This might need refinement based on strictness requirements.
    if server_output_modes is None or not server_output_modes:
         # If client ONLY accepts non-text, and server defaults to text, it's incompatible.
         if all(mode != "text" for mode in client_output_modes):
              return False
         return True # Assume server can produce 'text' if not specified

    # Check for any overlap between the two lists
    return any(mode in server_output_modes for mode in client_output_modes)


def new_incompatible_types_error(request_id: str | int | None) -> JSONRPCResponse:
    """Creates a standard JSON-RPC response for content type incompatibility."""
    return JSONRPCResponse(id=request_id, error=ContentTypeNotSupportedError())


def new_not_implemented_error(request_id: str | int | None) -> JSONRPCResponse:
    """Creates a standard JSON-RPC response for unsupported operations."""
    return JSONRPCResponse(id=request_id, error=UnsupportedOperationError())

```

### Task Manager (`task_manager.py`)

An abstract base class (`TaskManager`) defining the interface for handling A2A task-related requests. It also includes a concrete `InMemoryTaskManager` implementation that stores task data in memory. Agent implementations will typically inherit from `InMemoryTaskManager` and override methods like `on_send_task` and `on_send_task_subscribe` to add their specific logic.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\server\task_manager.py
from abc import ABC, abstractmethod
from typing import Union, AsyncIterable, List, Optional
from common.types import Task # Core Task model
from common.types import (
    JSONRPCResponse,
    TaskIdParams,
    TaskQueryParams,
    GetTaskRequest,
    TaskNotFoundError,
    SendTaskRequest,
    CancelTaskRequest,
    TaskNotCancelableError,
    SetTaskPushNotificationRequest,
    GetTaskPushNotificationRequest,
    GetTaskResponse,
    CancelTaskResponse,
    SendTaskResponse,
    SetTaskPushNotificationResponse,
    GetTaskPushNotificationResponse,
    PushNotificationNotSupportedError, # Error if push notifications aren't enabled/supported
    TaskSendParams,
    TaskStatus,
    TaskState,
    TaskResubscriptionRequest,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Artifact,
    PushNotificationConfig,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent, # Added for completeness
    JSONRPCError,
    TaskPushNotificationConfig,
    InternalError,
)
from common.server.utils import new_not_implemented_error # Utility for standard errors
import asyncio
import logging

logger = logging.getLogger(__name__)

class TaskManager(ABC):
    """Abstract Base Class defining the interface for handling A2A task operations.

    Agent implementations should inherit from this (or a concrete subclass like InMemoryTaskManager)
    and implement the required methods.
    """
    @abstractmethod
    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        """Handles 'tasks/get' requests."""
        pass

    @abstractmethod
    async def on_cancel_task(self, request: CancelTaskRequest) -> CancelTaskResponse:
        """Handles 'tasks/cancel' requests."""
        pass

    @abstractmethod
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles 'tasks/send' requests (non-streaming)."""
        pass

    @abstractmethod
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """Handles 'tasks/sendSubscribe' requests (streaming via SSE).

        Should return an async iterable yielding SendTaskStreamingResponse objects
        or a single JSONRPCResponse in case of immediate error.
        """
        pass

    @abstractmethod
    async def on_set_task_push_notification(
        self, request: SetTaskPushNotificationRequest
    ) -> SetTaskPushNotificationResponse:
        """Handles 'tasks/pushNotification/set' requests."""
        pass

    @abstractmethod
    async def on_get_task_push_notification(
        self, request: GetTaskPushNotificationRequest
    ) -> GetTaskPushNotificationResponse:
        """Handles 'tasks/pushNotification/get' requests."""
        pass

    @abstractmethod
    async def on_resubscribe_to_task(
        self, request: TaskResubscriptionRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]: # Changed return type to match subscribe
        """Handles 'tasks/resubscribe' requests.

        Should return an async iterable yielding SendTaskStreamingResponse objects
        representing the current state and future updates, or a single JSONRPCResponse
        in case of error (e.g., task not found).
        """
        pass


class InMemoryTaskManager(TaskManager):
    """A concrete TaskManager implementation storing task data in memory.

    Provides basic storage and retrieval logic. Agent implementations typically
    inherit from this and override `on_send_task` and `on_send_task_subscribe`
    to execute their specific task logic.
    """
    def __init__(self, supports_push_notifications: bool = False): # Added flag
        self.tasks: dict[str, Task] = {} # Stores Task objects by ID
        self.push_notification_configs: dict[str, PushNotificationConfig] = {} # Stores push config by Task ID
        self.lock = asyncio.Lock() # Lock for accessing shared task/push config data
        self.task_sse_subscribers: dict[str, List[asyncio.Queue]] = {} # Stores SSE queues by Task ID
        self.subscriber_lock = asyncio.Lock() # Lock specifically for managing subscriber queues
        self._supports_push_notifications = supports_push_notifications
        logger.info(f"InMemoryTaskManager initialized (Push Notifications Supported: {self._supports_push_notifications})")


    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        """Retrieves a task by ID from the in-memory store."""
        logger.info(f"Handling tasks/get for task {request.params.id}")
        task_query_params: TaskQueryParams = request.params

        async with self.lock:
            task = self.tasks.get(task_query_params.id)
            if task is None:
                logger.warning(f"Task {request.params.id} not found for tasks/get.")
                return GetTaskResponse(id=request.id, error=TaskNotFoundError())

            # Create a copy and potentially trim history before returning
            task_result = self._prepare_task_for_response(
                task, task_query_params.historyLength
            )

        return GetTaskResponse(id=request.id, result=task_result)

    async def on_cancel_task(self, request: CancelTaskRequest) -> CancelTaskResponse:
        """Handles task cancellation requests. Base implementation returns 'not cancelable'."""
        logger.info(f"Handling tasks/cancel for task {request.params.id}")
        task_id_params: TaskIdParams = request.params

        async with self.lock:
            task = self.tasks.get(task_id_params.id)
            if task is None:
                logger.warning(f"Task {request.params.id} not found for tasks/cancel.")
                return CancelTaskResponse(id=request.id, error=TaskNotFoundError())

            # TODO: Implement actual cancellation logic if possible
            # - Check if task is in a cancelable state (e.g., WORKING)
            # - Signal the running task to stop
            # - Update task status to CANCELED
            # For now, just return the error
            logger.warning(f"Task {request.params.id} cancellation not implemented/supported.")
            return CancelTaskResponse(id=request.id, error=TaskNotCancelableError())

    @abstractmethod
    async def _execute_task_logic(self, task: Task, task_send_params: TaskSendParams):
         """Placeholder for the actual agent logic.

         This method should be implemented by subclasses to perform the agent's work.
         It should use `update_task_status` and `add_task_artifact` to report progress.
         """
         pass

    async def _process_send_task_base(self, request_id: str | int, task_send_params: TaskSendParams, is_streaming: bool):
        """Common logic for both send_task and send_task_subscribe."""
        logger.info(f"Processing send task {'(streaming)' if is_streaming else ''} for task {task_send_params.id}, session {task_send_params.sessionId}")

        # 1. Upsert Task (create if new, add message to history if existing)
        try:
            task = await self.upsert_task(task_send_params)
        except Exception as e:
            logger.exception(f"Error upserting task {task_send_params.id}")
            return JSONRPCResponse(id=request_id, error=InternalError(message=f"Failed to initialize task: {e}"))

        # 2. Handle Push Notification Setup (if requested and supported)
        if task_send_params.pushNotification:
            if not self._supports_push_notifications:
                logger.warning(f"Push notification requested for task {task.id}, but not supported by this manager.")
                return JSONRPCResponse(id=request_id, error=PushNotificationNotSupportedError())
            try:
                # Store the config - verification should happen elsewhere (e.g., before sending)
                await self.set_push_notification_config(task.id, task_send_params.pushNotification)
                logger.info(f"Stored push notification config for task {task.id}")
            except Exception as e:
                 logger.exception(f"Error storing push notification config for task {task.id}")
                 return JSONRPCResponse(id=request_id, error=InternalError(message=f"Failed to store push notification config: {e}"))

        # 3. Setup SSE Queue if streaming
        sse_event_queue = None
        if is_streaming:
            try:
                sse_event_queue = await self.setup_sse_consumer(task.id)
            except Exception as e:
                 logger.exception(f"Error setting up SSE consumer for task {task.id}")
                 return JSONRPCResponse(id=request_id, error=InternalError(message=f"Failed to set up streaming: {e}"))

        # 4. Start Task Execution in Background
        async def task_execution_wrapper():
            try:
                await self._execute_task_logic(task, task_send_params)
            except Exception as e:
                logger.exception(f"Error during task execution for {task.id}")
                # Report failure status
                await self.update_task_status(
                    task.id,
                    TaskStatus(state=TaskState.FAILED, message=TextPart(text=f"Task failed: {e}").model_dump()),
                    final=True
                )
            finally:
                 # Ensure final event is sent if streaming and task ends without explicit final status
                 if is_streaming and sse_event_queue:
                      # Check last status before potentially sending another final event
                      async with self.lock:
                           final_status = self.tasks[task.id].status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]
                      if not final_status:
                           logger.warning(f"Task {task.id} execution finished without a final status. Setting to FAILED.")
                           await self.update_task_status(
                                task.id,
                                TaskStatus(state=TaskState.FAILED, message=TextPart(text="Task ended unexpectedly.").model_dump()),
                                final=True
                           )


        asyncio.create_task(task_execution_wrapper())
        logger.info(f"Task {task.id} execution started in background.")

        # 5. Return appropriate response
        if is_streaming:
            # Return the async generator that reads from the queue
            return self.dequeue_events_for_sse(request_id, task.id, sse_event_queue)
        else:
            # Return the initial task state for non-streaming
            async with self.lock:
                initial_task_state = self._prepare_task_for_response(task, task_send_params.historyLength)
            return SendTaskResponse(id=request_id, result=initial_task_state)


    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles non-streaming task requests."""
        result = await self._process_send_task_base(request.id, request.params, is_streaming=False)
        if isinstance(result, JSONRPCResponse): # Handle immediate errors
             return SendTaskResponse(id=request.id, error=result.error)
        return result # Should be SendTaskResponse


    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """Handles streaming task requests."""
        return await self._process_send_task_base(request.id, request.params, is_streaming=True)


    async def set_push_notification_config(self, task_id: str, notification_config: PushNotificationConfig):
        """Stores the push notification configuration for a task."""
        if not self._supports_push_notifications:
             raise PushNotificationNotSupportedError()

        async with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task not found for {task_id}") # Or TaskNotFoundError?
            # TODO: Potentially verify the URL here if not done elsewhere
            self.push_notification_configs[task_id] = notification_config
        logger.info(f"Push notification config set for task {task_id}")


    async def get_push_notification_config(self, task_id: str) -> PushNotificationConfig | None:
         """Retrieves the push notification configuration for a task."""
         if not self._supports_push_notifications:
              return None # Or raise error?
         async with self.lock:
              return self.push_notification_configs.get(task_id)


    async def on_set_task_push_notification(
        self, request: SetTaskPushNotificationRequest
    ) -> SetTaskPushNotificationResponse:
        """Handles requests to set push notification config."""
        logger.info(f"Handling tasks/pushNotification/set for task {request.params.id}")
        if not self._supports_push_notifications:
            logger.warning("Attempted to set push notification, but not supported.")
            return SetTaskPushNotificationResponse(id=request.id, error=PushNotificationNotSupportedError())

        task_notification_params: TaskPushNotificationConfig = request.params
        try:
            await self.set_push_notification_config(task_notification_params.id, task_notification_params.pushNotificationConfig)
            # Return the config that was set
            return SetTaskPushNotificationResponse(id=request.id, result=task_notification_params)
        except ValueError as e: # Catch task not found
             logger.warning(f"Task {request.params.id} not found for setting push notification.")
             return SetTaskPushNotificationResponse(id=request.id, error=TaskNotFoundError(message=str(e)))
        except Exception as e:
            logger.exception(f"Error setting push notification config for task {request.params.id}")
            return SetTaskPushNotificationResponse(id=request.id, error=InternalError(message=f"Failed to set push notification config: {e}"))


    async def on_get_task_push_notification(
        self, request: GetTaskPushNotificationRequest
    ) -> GetTaskPushNotificationResponse:
        """Handles requests to get push notification config."""
        logger.info(f"Handling tasks/pushNotification/get for task {request.params.id}")
        if not self._supports_push_notifications:
             logger.warning("Attempted to get push notification, but not supported.")
             return GetTaskPushNotificationResponse(id=request.id, error=PushNotificationNotSupportedError())

        task_params: TaskIdParams = request.params
        try:
            async with self.lock:
                if task_params.id not in self.tasks:
                     logger.warning(f"Task {request.params.id} not found for getting push notification.")
                     return GetTaskPushNotificationResponse(id=request.id, error=TaskNotFoundError())

                notification_config = self.push_notification_configs.get(task_params.id)
                if notification_config is None:
                     # Spec is unclear if this is an error or just returns null/empty result.
                     # Let's return null result for now.
                     logger.info(f"No push notification config found for task {task_params.id}.")
                     return GetTaskPushNotificationResponse(id=request.id, result=None) # Or TaskPushNotificationConfig with null?

            result_config = TaskPushNotificationConfig(id=task_params.id, pushNotificationConfig=notification_config)
            return GetTaskPushNotificationResponse(id=request.id, result=result_config)
        except Exception as e:
            logger.exception(f"Error getting push notification config for task {request.params.id}")
            return GetTaskPushNotificationResponse(id=request.id, error=InternalError(message=f"Failed to get push notification config: {e}"))


    async def upsert_task(self, task_send_params: TaskSendParams) -> Task:
        """Creates a new task or adds a message to an existing task's history."""
        logger.info(f"Upserting task {task_send_params.id} for session {task_send_params.sessionId}")
        async with self.lock:
            task = self.tasks.get(task_send_params.id)
            if task is None:
                # Create new task
                task = Task(
                    id=task_send_params.id,
                    sessionId=task_send_params.sessionId,
                    # messages field seems deprecated/unused in favor of history
                    status=TaskStatus(state=TaskState.SUBMITTED), # Initial state
                    history=[task_send_params.message], # Start history with the first message
                    artifacts=[], # Initialize artifacts list
                    metadata=task_send_params.metadata,
                )
                self.tasks[task_send_params.id] = task
                logger.info(f"Created new task {task.id}")
            else:
                # Add message to existing task history
                if task.history is None: # Should not happen if initialized correctly
                     task.history = []
                task.history.append(task_send_params.message)
                # Optionally update metadata if provided? Spec is unclear.
                # task.metadata = task_send_params.metadata or task.metadata
                logger.info(f"Appended message to existing task {task.id}")

            return task.model_copy(deep=True) # Return a copy to avoid external modification


    async def on_resubscribe_to_task(
        self, request: TaskResubscriptionRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """Handles resubscription requests. Base implementation returns 'not implemented'."""
        logger.warning(f"Task resubscription requested for {request.params.id}, but not implemented.")
        # TODO: Implement resubscription
        # 1. Find task
        # 2. Check if task is still active/streamable
        # 3. Create a new SSE queue
        # 4. Send current status and recent artifacts/history to the new queue
        # 5. Add queue to subscribers list
        # 6. Return the dequeue generator
        return new_not_implemented_error(request.id)


    async def update_task_status(self, task_id: str, status: TaskStatus, final: bool = False):
        """Updates the status of a task and notifies subscribers/push endpoints."""
        logger.info(f"Updating task {task_id} status to {status.state} (Final: {final})")
        async with self.lock:
            try:
                task = self.tasks[task_id]
            except KeyError:
                logger.error(f"Task {task_id} not found for status update.")
                # Cannot proceed if task doesn't exist
                return

            task.status = status
            if status.message: # Add agent messages to history
                if task.history is None: task.history = []
                task.history.append(status.message)

        # Create event payload
        event = TaskStatusUpdateEvent(id=task_id, status=status, final=final)

        # Enqueue for SSE subscribers
        await self.enqueue_events_for_sse(task_id, event)

        # Send push notification (if configured and applicable)
        # Push notifications might be sent only for final states or specific transitions
        # depending on desired behavior. Let's assume push for final states for now.
        if final and self._supports_push_notifications:
             push_config = self.push_notification_configs.get(task_id)
             if push_config:
                  # Need access to the PushNotificationSenderAuth instance here
                  # This suggests the TaskManager might need a reference to it,
                  # or this logic needs to be called from a place that has it.
                  # For now, just log the intent.
                  logger.info(f"Intending to send push notification for final status of task {task_id} to {push_config.url}")
                  # await push_notification_sender.send_push_notification(push_config.url, event.model_dump())
             else:
                  logger.debug(f"No push notification config found for final status of task {task_id}")


    async def add_task_artifact(self, task_id: str, artifact: Artifact):
        """Adds an artifact to a task and notifies subscribers/push endpoints."""
        logger.info(f"Adding artifact (Index: {artifact.index}, Last: {artifact.lastChunk}) to task {task_id}")
        async with self.lock:
            try:
                task = self.tasks[task_id]
            except KeyError:
                logger.error(f"Task {task_id} not found for adding artifact.")
                return

            if task.artifacts is None:
                task.artifacts = []

            # Handle artifact chunking/appending (basic implementation)
            # Assumes index is sequential for appending. A more robust implementation
            # might store chunks separately and reconstruct later.
            if artifact.append and task.artifacts and task.artifacts[-1].index == artifact.index:
                 # Append parts to the last artifact at the same index
                 task.artifacts[-1].parts.extend(artifact.parts)
                 task.artifacts[-1].lastChunk = artifact.lastChunk # Update lastChunk status
                 logger.debug(f"Appended parts to artifact index {artifact.index} for task {task_id}")
            else:
                 # Add as a new artifact entry (or replace if index exists?)
                 # For simplicity, let's just append. Handling replacement/updates is complex.
                 task.artifacts.append(artifact)
                 logger.debug(f"Added new artifact entry index {artifact.index} for task {task_id}")


        # Create event payload
        event = TaskArtifactUpdateEvent(id=task_id, artifact=artifact)

        # Enqueue for SSE subscribers
        await self.enqueue_events_for_sse(task_id, event)

        # Send push notification (if configured and applicable)
        # Typically, push notifications might be sent less frequently for artifacts,
        # maybe only on lastChunk=True or based on specific metadata.
        if artifact.lastChunk and self._supports_push_notifications:
             push_config = self.push_notification_configs.get(task_id)
             if push_config:
                  logger.info(f"Intending to send push notification for final artifact chunk of task {task_id} to {push_config.url}")
                  # await push_notification_sender.send_push_notification(push_config.url, event.model_dump())


    def _prepare_task_for_response(self, task: Task, history_length: int | None) -> Task:
        """Creates a copy of the task suitable for returning in a response.

        Specifically handles trimming the history based on the requested length.
        """
        response_task = task.model_copy(deep=True) # Work with a copy

        if response_task.history:
            if history_length is None or history_length < 0:
                # Negative or None means no history
                response_task.history = []
            elif history_length > 0:
                # Return the last 'history_length' messages
                response_task.history = response_task.history[-history_length:]
            # history_length == 0 means return empty list (already handled by default slicing if needed)
        else:
             response_task.history = [] # Ensure it's an empty list if None initially

        # Potentially filter artifacts or other fields based on request params in the future
        return response_task


    async def setup_sse_consumer(self, task_id: str, is_resubscribe: bool = False) -> asyncio.Queue:
        """Sets up a queue for a new SSE subscriber for a given task."""
        async with self.subscriber_lock:
            if task_id not in self.task_sse_subscribers:
                if is_resubscribe:
                    logger.error(f"Task {task_id} not found for SSE resubscription.")
                    raise ValueError("Task not found for resubscription")
                else:
                    # Initialize list for this task ID if it's the first subscriber
                    self.task_sse_subscribers[task_id] = []
                    logger.info(f"Initialized SSE subscriber list for task {task_id}")

            # Create a new queue for this specific subscriber connection
            sse_event_queue = asyncio.Queue(maxsize=0) # Unlimited size queue
            self.task_sse_subscribers[task_id].append(sse_event_queue)
            logger.info(f"Added new SSE subscriber queue for task {task_id}. Total subscribers: {len(self.task_sse_subscribers[task_id])}")
            return sse_event_queue

    async def enqueue_events_for_sse(self, task_id: str, event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent, JSONRPCError]):
        """Puts an event into the queues of all active SSE subscribers for a task."""
        async with self.subscriber_lock:
            if task_id not in self.task_sse_subscribers:
                # No active subscribers for this task
                return

            subscribers = self.task_sse_subscribers[task_id]
            if not subscribers:
                 # List might exist but be empty if subscribers disconnected
                 return

            logger.debug(f"Enqueuing event for {len(subscribers)} SSE subscribers of task {task_id}")
            # Put the event into each subscriber's queue
            # Use asyncio.gather for slight potential concurrency, though queue.put is usually fast
            await asyncio.gather(*(subscriber.put(event) for subscriber in subscribers))


    async def dequeue_events_for_sse(
        self, request_id: str | int, task_id: str, sse_event_queue: asyncio.Queue
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        """Async generator that yields events from a specific SSE subscriber queue."""
        logger.info(f"Starting SSE event dequeuing for task {task_id}, request {request_id}")
        try:
            while True:
                # Wait for an event from this subscriber's queue
                event = await sse_event_queue.get()
                logger.debug(f"Dequeued event for task {task_id}, request {request_id}: {type(event)}")

                if isinstance(event, JSONRPCError):
                    # If an error object was put in the queue, yield it and stop
                    yield SendTaskStreamingResponse(id=request_id, error=event)
                    logger.warning(f"Error event dequeued for task {task_id}, request {request_id}. Stopping stream.")
                    break
                elif isinstance(event, (TaskStatusUpdateEvent, TaskArtifactUpdateEvent)):
                    # Yield the actual status or artifact update
                    yield SendTaskStreamingResponse(id=request_id, result=event)
                    # Check if it's a final status update
                    if isinstance(event, TaskStatusUpdateEvent) and event.final:
                        logger.info(f"Final status event dequeued for task {task_id}, request {request_id}. Stopping stream.")
                        break # Stop the generator on final status
                else:
                     # Should not happen if only valid events are enqueued
                     logger.error(f"Unknown event type dequeued for task {task_id}: {type(event)}")
                     yield SendTaskStreamingResponse(id=request_id, error=InternalError(message="Unknown event type received"))
                     break

                sse_event_queue.task_done() # Mark task as done for the queue

        except asyncio.CancelledError:
             logger.info(f"SSE dequeuing cancelled for task {task_id}, request {request_id}.")
             # Propagate cancellation if needed
             raise
        except Exception as e:
             logger.exception(f"Unexpected error during SSE dequeuing for task {task_id}, request {request_id}")
             # Yield an internal error before stopping
             yield SendTaskStreamingResponse(id=request_id, error=InternalError(message=f"Dequeuing error: {e}"))
        finally:
            # Clean up: remove this queue from the list of subscribers for the task
            async with self.subscriber_lock:
                if task_id in self.task_sse_subscribers:
                    try:
                        self.task_sse_subscribers[task_id].remove(sse_event_queue)
                        logger.info(f"Removed SSE subscriber queue for task {task_id}, request {request_id}. Remaining: {len(self.task_sse_subscribers[task_id])}")
                        # Optional: clean up task entry if no subscribers left and task is final?
                        # if not self.task_sse_subscribers[task_id] and task is final:
                        #     del self.task_sse_subscribers[task_id]
                        #     # Maybe remove task from self.tasks too after a delay?
                    except ValueError:
                        # Queue might have already been removed (e.g., race condition) - ignore
                        logger.warning(f"SSE subscriber queue for task {task_id}, request {request_id} already removed.")
                        pass
```

### A2A Server (`server.py`)

A Starlette-based web server that exposes the A2A endpoints (`/` for API calls, `/.well-known/agent.json` for the card). It receives JSON-RPC requests, routes them to the appropriate `TaskManager` method, and handles response formatting (JSON or SSE) and error handling.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\server\server.py
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse # For Server-Sent Events
from starlette.requests import Request
from starlette.routing import Route, Mount # For defining routes
from starlette.middleware import Middleware # If middleware is needed
# from starlette.middleware.cors import CORSMiddleware # Example middleware

from common.types import (
    A2ARequest, # Type adapter for parsing incoming requests
    JSONRPCResponse,
    InvalidRequestError, # Specific JSON-RPC errors
    JSONParseError,
    MethodNotFoundError, # Although handled by A2ARequest adapter
    GetTaskRequest,
    CancelTaskRequest,
    SendTaskRequest,
    SetTaskPushNotificationRequest,
    GetTaskPushNotificationRequest,
    InternalError,
    AgentCard,
    TaskResubscriptionRequest,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse, # Type used in streaming
)
from pydantic import ValidationError # For catching Pydantic validation errors
import json
from typing import AsyncIterable, Any, Union
from common.server.task_manager import TaskManager # The core task handling logic

import logging
import traceback # For logging stack traces

logger = logging.getLogger(__name__)


class A2AServer:
    """A Starlette-based web server for hosting an A2A agent."""
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        endpoint: str = "/", # Base path for the A2A API
        agent_card: AgentCard = None,
        task_manager: TaskManager = None,
        # Optional: Add push notification sender if supported
        # push_notification_sender: PushNotificationSenderAuth = None,
        middleware: list = None, # Allow custom middleware
    ):
        """Initializes the A2A server.

        Args:
            host: Host address to bind to.
            port: Port number to listen on.
            endpoint: The API endpoint path.
            agent_card: The AgentCard object describing this agent.
            task_manager: The TaskManager instance handling task logic.
            middleware: Optional list of Starlette middleware.
        """
        self.host = host
        self.port = port
        self.endpoint = endpoint.rstrip('/') # Ensure no trailing slash
        self.task_manager = task_manager
        self.agent_card = agent_card
        # self.push_notification_sender = push_notification_sender # Store if provided

        if not self.agent_card:
            raise ValueError("agent_card must be provided")
        if not self.task_manager:
            raise ValueError("task_manager must be provided")

        # Define routes
        routes = [
            Route(self.endpoint, self._process_request, methods=["POST"], name="a2a_api"),
            Route("/.well-known/agent.json", self._get_agent_card, methods=["GET"], name="agent_card"),
        ]

        # Add JWKS endpoint if push notifications are supported by the task manager
        # This assumes the task manager holds or has access to the sender auth logic
        # A cleaner approach might be to pass the sender_auth object directly.
        if hasattr(self.task_manager, '_supports_push_notifications') and self.task_manager._supports_push_notifications:
             if hasattr(self.task_manager, 'push_notification_sender') and self.task_manager.push_notification_sender:
                  routes.append(Route(
                       "/.well-known/jwks.json",
                       self.task_manager.push_notification_sender.handle_jwks_endpoint,
                       methods=["GET"],
                       name="jwks"
                  ))
                  logger.info("JWKS endpoint enabled at /.well-known/jwks.json")
             else:
                  logger.warning("Task manager supports push notifications but has no 'push_notification_sender' attribute/object.")


        # Initialize Starlette app with routes and optional middleware
        self.app = Starlette(routes=routes, middleware=middleware)

        logger.info(f"A2AServer initialized. API endpoint: {self.endpoint}, Agent Card: /.well-known/agent.json")


    def start(self):
        """Starts the Uvicorn server to run the Starlette application."""
        # Basic validation already done in __init__
        import uvicorn

        logger.info(f"Starting A2AServer on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)

    def _get_agent_card(self, request: Request) -> JSONResponse:
        """Handles GET requests for the agent card."""
        logger.debug(f"Serving Agent Card request from {request.client.host}")
        return JSONResponse(self.agent_card.model_dump(exclude_none=True))

    async def _process_request(self, request: Request) -> Union[JSONResponse, EventSourceResponse]:
        """Processes incoming POST requests to the main API endpoint."""
        request_id = None # Keep track of request ID for error responses
        try:
            # 1. Parse request body as JSON
            try:
                body = await request.json()
                # Attempt to get request ID early for error reporting
                request_id = body.get("id")
                logger.debug(f"Received request body for ID {request_id}: {body}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse request JSON: {e}")
                raise JSONParseError(message=f"Invalid JSON payload: {e}") from e # Raise specific error

            # 2. Validate and parse using the A2ARequest TypeAdapter
            try:
                # This adapter uses the 'method' field to determine the correct Pydantic model
                json_rpc_request = A2ARequest.validate_python(body)
                request_id = json_rpc_request.id # Update request_id from validated request
                logger.info(f"Processing method '{json_rpc_request.method}' for request ID {request_id}")
            except ValidationError as e:
                logger.warning(f"Request validation failed: {e.errors()}")
                # Provide validation details in the error data
                raise InvalidRequestError(data=e.errors()) from e # Raise specific error

            # 3. Route to the appropriate TaskManager method
            result: Union[JSONRPCResponse, AsyncIterable[SendTaskStreamingResponse]]

            if isinstance(json_rpc_request, GetTaskRequest):
                result = await self.task_manager.on_get_task(json_rpc_request)
            elif isinstance(json_rpc_request, SendTaskRequest):
                result = await self.task_manager.on_send_task(json_rpc_request)
            elif isinstance(json_rpc_request, SendTaskStreamingRequest):
                result = await self.task_manager.on_send_task_subscribe(json_rpc_request)
            elif isinstance(json_rpc_request, CancelTaskRequest):
                result = await self.task_manager.on_cancel_task(json_rpc_request)
            elif isinstance(json_rpc_request, SetTaskPushNotificationRequest):
                result = await self.task_manager.on_set_task_push_notification(json_rpc_request)
            elif isinstance(json_rpc_request, GetTaskPushNotificationRequest):
                result = await self.task_manager.on_get_task_push_notification(json_rpc_request)
            elif isinstance(json_rpc_request, TaskResubscriptionRequest):
                result = await self.task_manager.on_resubscribe_to_task(json_rpc_request)
            else:
                # Should not happen if A2ARequest adapter covers all methods
                logger.error(f"Unhandled request type after validation: {type(json_rpc_request)}")
                raise InternalError(message=f"Unexpected request type: {type(json_rpc_request)}")

            # 4. Create and return the appropriate response (JSON or SSE)
            return self._create_response(result, request_id) # Pass request_id for context

        except Exception as e:
            # Catch any exceptions raised during parsing, validation, or handler execution
            return self._handle_exception(e, request_id)


    def _handle_exception(self, e: Exception, request_id: Any = None) -> JSONResponse:
        """Handles exceptions and converts them into standard JSON-RPC error responses."""
        json_rpc_error: JSONRPCError

        if isinstance(e, JSONParseError):
            json_rpc_error = e
        elif isinstance(e, InvalidRequestError):
            json_rpc_error = e
        elif isinstance(e, (TaskNotFoundError, TaskNotCancelableError, PushNotificationNotSupportedError, UnsupportedOperationError, ContentTypeNotSupportedError)):
             # Handle specific A2A errors raised by the TaskManager
             json_rpc_error = e
        else:
            # Catch-all for unexpected internal errors
            logger.exception(f"Unhandled internal server error for request ID {request_id}: {e}") # Log full traceback
            json_rpc_error = InternalError(data={"exception_type": type(e).__name__, "detail": str(e)})

        # Create a JSON-RPC response containing the error
        response = JSONRPCResponse(id=request_id, error=json_rpc_error)
        status_code = 400 # Default bad request for most JSON-RPC errors
        if isinstance(json_rpc_error, InternalError):
             status_code = 500 # Internal server error

        return JSONResponse(
            response.model_dump(exclude_none=True),
            status_code=status_code
        )

    def _create_response(self, result: Any, request_id: Any) -> Union[JSONResponse, EventSourceResponse]:
        """Creates a JSONResponse or EventSourceResponse based on the handler result."""

        # Handle streaming responses (SSE)
        if isinstance(result, AsyncIterable):
            logger.info(f"Creating SSE response for request ID {request_id}")

            async def event_generator(stream: AsyncIterable[SendTaskStreamingResponse]) -> AsyncIterable[dict[str, str]]:
                try:
                    async for item in stream:
                        # Each item from the stream should be a SendTaskStreamingResponse
                        # We serialize its model_dump_json to send as SSE data
                        yield {"event": "message", "data": item.model_dump_json(exclude_none=True)}
                except Exception as e:
                     logger.exception(f"Error during SSE event generation for request ID {request_id}")
                     # Yield a final error event if possible
                     error_response = SendTaskStreamingResponse(id=request_id, error=InternalError(message=f"SSE generation error: {e}"))
                     yield {"event": "error", "data": error_response.model_dump_json(exclude_none=True)}


            return EventSourceResponse(event_generator(result))

        # Handle single JSON-RPC responses
        elif isinstance(result, JSONRPCResponse):
            logger.info(f"Creating JSON response for request ID {request_id} (Success: {result.error is None})")
            status_code = 200
            if result.error:
                 # Determine appropriate HTTP status code based on JSON-RPC error code
                 if result.error.code in [-32700, -32600, -32601, -32602, -32005]: # Parse, Invalid Req, Method/Params, ContentType
                      status_code = 400
                 elif result.error.code == -32001: # Task Not Found
                      status_code = 404 # Not Found might be suitable
                 elif result.error.code == -32603: # Internal Error
                      status_code = 500
                 # Add more mappings as needed
                 else:
                      status_code = 400 # Default bad request for other errors

            return JSONResponse(result.model_dump(exclude_none=True), status_code=status_code)

        # Handle unexpected result types
        else:
            logger.error(f"TaskManager returned unexpected result type: {type(result)} for request ID {request_id}")
            # Raise an internal error that will be caught and handled
            raise InternalError(message=f"Internal error: Unexpected result type from handler: {type(result)}")

```

### Server Init (`__init__.py`)

Makes the core server classes easily importable from the `common.server` package.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\common\server\__init__.py
from .server import A2AServer
from .task_manager import TaskManager, InMemoryTaskManager

# Expose the main classes for easier import
__all__ = ["A2AServer", "TaskManager", "InMemoryTaskManager"]
```

---

## 5. Sample CLI Host (`hosts/cli/`)

A command-line interface application demonstrating how to use the `A2AClient` to interact with an agent.

### Push Notification Listener (`push_notification_listener.py`)

A simple Starlette server run in a separate thread to listen for incoming push notifications from the agent. It uses the `PushNotificationReceiverAuth` utility to verify the notifications.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\hosts\cli\push_notification_listener.py
import asyncio
import threading
import logging
import traceback # For detailed error logging

# Import the receiver authentication utility
from common.utils.push_notification_auth import PushNotificationReceiverAuth

# Starlette components for building the web server
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, JSONResponse # Added JSONResponse for errors
from starlette.routing import Route

# Uvicorn for running the Starlette app
import uvicorn

logger = logging.getLogger(__name__)

class PushNotificationListener():
    """Runs a simple Starlette server in a background thread to listen for push notifications."""
    def __init__(self, host: str, port: int, notification_receiver_auth: PushNotificationReceiverAuth):
        """
        Args:
            host: Host address to bind the listener server to.
            port: Port number for the listener server.
            notification_receiver_auth: An initialized PushNotificationReceiverAuth instance
                                         (already loaded with the agent's JWKS URL).
        """
        self.host = host
        self.port = port
        self.notification_receiver_auth = notification_receiver_auth
        self.server: uvicorn.Server | None = None # To hold the Uvicorn server instance

        # Setup asyncio event loop for the background thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_event_loop, args=(self.loop,), daemon=True
        )
        self.thread.start()
        logger.info("Push notification listener thread started.")

    def _run_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Target function for the background thread to run the asyncio event loop."""
        logger.info("Background event loop started.")
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()
            logger.info("Background event loop stopped and closed.")

    def start(self):
        """Starts the Uvicorn server in the background thread's event loop."""
        try:
            # Schedule the server startup coroutine in the background loop
            asyncio.run_coroutine_threadsafe(
                self._start_server(),
                self.loop,
            )
            logger.info(f"Push notification listener scheduled to start on http://{self.host}:{self.port}")
        except Exception as e:
            logger.exception("Failed to schedule push notification listener startup.")

    async def _start_server(self):
        """Coroutine that sets up and runs the Uvicorn server."""
        routes = [
            Route("/notify", self.handle_notification, methods=["POST"]),
            # Endpoint for agent to verify the listener URL (Webhook validation)
            Route("/notify", self.handle_validation_check, methods=["GET"]),
        ]
        self.app = Starlette(routes=routes)

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info") # Use info level for uvicorn
        self.server = uvicorn.Server(config)
        logger.info(f"Uvicorn server starting on http://{self.host}:{self.port}/notify")
        try:
            await self.server.serve()
        except Exception as e:
             logger.exception("Uvicorn server encountered an error.")
        finally:
             logger.info("Uvicorn server stopped.")


    async def handle_validation_check(self, request: Request) -> Response:
        """Handles GET requests for webhook validation.

        Checks for a 'validationToken' query parameter and returns it in the response body
        if found, with a 200 status. This allows the agent (sender) to confirm the endpoint.
        """
        validation_token = request.query_params.get("validationToken")
        if not validation_token:
            logger.warning("Validation check received without validationToken.")
            return Response("Missing validationToken", status_code=400)

        logger.info(f"Received validation check with token: {validation_token}")
        # Return the token in the response body with 200 OK
        return Response(content=validation_token, status_code=200, media_type="text/plain")


    async def handle_notification(self, request: Request) -> Response:
        """Handles incoming POST requests containing push notifications."""
        logger.info("Received POST request on /notify")
        try:
            # 1. Verify the request using the authentication utility
            is_verified = await self.notification_receiver_auth.verify_push_notification(request)

            if not is_verified:
                logger.warning("Push notification verification failed.")
                # Return 401 Unauthorized or 403 Forbidden
                return JSONResponse({"error": "Verification failed"}, status_code=401)

            # 2. Process the verified notification data
            # Need to re-read body if verify_push_notification consumed it
            # Note: verify_push_notification currently reads the body. If it needs to be
            # read again, the verification logic might need adjustment, or Starlette's
            # Request object needs careful handling (e.g., caching the body).
            # Assuming verify_push_notification doesn't consume it irreversibly for now,
            # or that we only need verification. Let's try reading it again.
            try:
                 data = await request.json()
                 logger.info(f"Verified push notification received: {json.dumps(data, indent=2)}")
                 # TODO: Add logic here to actually *do* something with the notification
                 # (e.g., update UI, store data, trigger another action)

            except json.JSONDecodeError:
                 logger.error("Failed to decode JSON from verified push notification.")
                 return JSONResponse({"error": "Invalid JSON body"}, status_code=400)


            # 3. Send success response
            return Response(status_code=200) # Simple 200 OK on success

        except Exception as e:
            # Log unexpected errors during verification or processing
            logger.exception("Error handling push notification.")
            # Return a 500 Internal Server Error response
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    def stop(self):
         """Stops the Uvicorn server gracefully."""
         if self.server and self.loop.is_running():
              logger.info("Requesting Uvicorn server shutdown.")
              # Schedule shutdown in the background loop
              asyncio.run_coroutine_threadsafe(self.server.shutdown(), self.loop)
              # Optionally stop the loop itself after shutdown completes
              # self.loop.call_soon_threadsafe(self.loop.stop)
         # Wait for the thread to finish? Depends on desired cleanup.
         # self.thread.join()

```

### CLI Main (`__main__.py`)

The main entry point for the CLI host. It uses `asyncclick` for async command-line argument parsing. It initializes the `A2AClient`, optionally starts the `PushNotificationListener`, and enters a loop prompting the user for input to send to the agent. It handles both streaming and non-streaming responses.

```python
# filepath: c:\Users\selhousseini\Documents\GitHub\A2A\samples\python\hosts\cli\__main__.py
import asyncclick as click # Async version of Click CLI library
import asyncio
import base64 # For encoding file content
import os
import urllib.parse # For parsing URLs
from uuid import uuid4
import json # For pretty printing
import logging

# A2A Client components
from common.client import A2AClient, A2ACardResolver
# A2A Types
from common.types import TaskState, Task, TextPart, FilePart, FileContent, Message # Added Message
# Push Notification utility
from common.utils.push_notification_auth import PushNotificationReceiverAuth

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing the listener only if needed
try:
    from hosts.cli.push_notification_listener import PushNotificationListener
    _listener_available = True
except ImportError:
    _listener_available = False
    logger.warning("PushNotificationListener not available. Push notifications disabled.")


@click.command()
@click.option("--agent", default="http://localhost:10000", help="Base URL of the A2A agent.")
@click.option("--session", default=None, help="Reuse an existing session ID.")
@click.option("--history", is_flag=True, default=False, help="Show task history after completion.")
@click.option("--use-push-notifications", is_flag=True, default=False, help="Enable and use push notifications.")
@click.option("--push-notification-receiver", default="http://localhost:5001", help="URL where the CLI should listen for push notifications.")
async def cli(agent: str, session: str | None, history: bool, use_push_notifications: bool, push_notification_receiver: str):
    """A command-line host application to interact with an A2A agent."""

    listener = None
    notification_receiver_auth = None

    # --- Agent Card Fetching ---
    try:
        logger.info(f"Fetching agent card from: {agent}")
        card_resolver = A2ACardResolver(agent)
        # Use async version if available, otherwise sync
        if hasattr(card_resolver, 'get_agent_card_async'):
             card = await card_resolver.get_agent_card_async()
        else:
             card = card_resolver.get_agent_card() # Fallback to sync

        click.echo("======= Agent Card ========")
        click.echo(card.model_dump_json(indent=2, exclude_none=True))
        click.echo("===========================")

    except Exception as e:
        logger.exception(f"Failed to fetch or parse agent card from {agent}.")
        click.echo(f"Error: Could not get agent card from {agent}. Details: {e}", err=True)
        return # Exit if card cannot be fetched

    # --- Push Notification Setup ---
    if use_push_notifications:
        if not _listener_available:
             click.echo("Error: Cannot use push notifications, listener component failed to import.", err=True)
             return
        if not card.capabilities.pushNotifications:
             click.echo(f"Warning: Agent at {agent} does not support push notifications (according to its card). Proceeding without them.", err=True)
             use_push_notifications = False # Disable if agent doesn't support it
        else:
            try:
                notif_receiver_parsed = urllib.parse.urlparse(push_notification_receiver)
                notification_receiver_host = notif_receiver_parsed.hostname
                notification_receiver_port = notif_receiver_parsed.port
                if not notification_receiver_host or not notification_receiver_port:
                     raise ValueError("Invalid push_notification_receiver URL format.")

                # Initialize receiver auth and load agent's public keys
                notification_receiver_auth = PushNotificationReceiverAuth()
                jwks_url = f"{agent}/.well-known/jwks.json"
                logger.info(f"Loading JWKS for push notification verification from: {jwks_url}")
                await notification_receiver_auth.load_jwks(jwks_url)

                # Start the listener server in background
                listener = PushNotificationListener(
                    host=notification_receiver_host,
                    port=notification_receiver_port,
                    notification_receiver_auth=notification_receiver_auth,
                )
                listener.start()
                click.echo(f"Push notification listener started at {push_notification_receiver}/notify")

            except Exception as e:
                logger.exception("Failed to set up push notification listener.")
                click.echo(f"Error setting up push notifications: {e}", err=True)
                use_push_notifications = False # Disable on setup failure

    # --- Initialize A2A Client ---
    try:
        client = A2AClient(agent_card=card) # Initialize client using the fetched card
    except Exception as e:
         logger.exception("Failed to initialize A2AClient.")
         click.echo(f"Error initializing A2A client: {e}", err=True)
         return

    # --- Session Management ---
    session_id = session if session else uuid4().hex # Use provided session or generate new
    click.echo(f"Using Session ID: {session_id}")

    # --- Main Interaction Loop ---
    streaming = card.capabilities.streaming # Check if agent supports streaming
    click.echo(f"Agent supports streaming: {streaming}")

    try:
        while True:
            task_id = uuid4().hex # New task ID for each interaction loop
            click.echo("========= Starting a new task ========")
            click.echo(f"Task ID: {task_id}")

            # Get user input and optional file
            prompt = await click.prompt(
                "What do you want to send to the agent? (Type :q or quit to exit)",
                type=str
            )
            if prompt.lower() in [":q", "quit"]:
                break

            file_path = await click.prompt(
                "Attach a file? (Enter path or press Enter to skip)",
                default="",
                show_default=False,
                type=str
            )

            # Construct the message payload
            message_parts = [TextPart(text=prompt)]
            if file_path and file_path.strip():
                try:
                    # Read file, encode, and add as FilePart
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                        file_content_b64 = base64.b64encode(file_bytes).decode('utf-8')
                        file_name = os.path.basename(file_path)
                        # TODO: Add MIME type detection if needed
                        message_parts.append(FilePart(file=FileContent(name=file_name, bytes=file_content_b64)))
                    click.echo(f"Attached file: {file_path}")
                except FileNotFoundError:
                    click.echo(f"Warning: File not found at '{file_path}'. Sending text only.", err=True)
                except Exception as e:
                    click.echo(f"Warning: Error reading file '{file_path}': {e}. Sending text only.", err=True)

            message = Message(role="user", parts=message_parts)

            # Construct the full task payload
            payload = {
                "id": task_id,
                "sessionId": session_id,
                "acceptedOutputModes": ["text"], # CLI primarily handles text output
                "message": message.model_dump(exclude_none=True), # Send message model as dict
            }

            # Add push notification config if enabled
            if use_push_notifications and listener:
                 # Ensure listener setup succeeded
                 payload["pushNotification"] = {
                      "url": f"http://{notification_receiver_host}:{notification_receiver_port}/notify",
                      "authentication": {
                           "schemes": ["bearer"], # Indicate JWT bearer token expected by listener
                      },
                 }
                 click.echo("Requesting push notifications.")


            # --- Send Task and Handle Response ---
            task_result: Task | None = None
            final_state: TaskState | None = None

            try:
                if streaming:
                    click.echo("--- Streaming Response ---")
                    response_stream = client.send_task_streaming(payload)
                    async for stream_event in response_stream:
                        click.echo(f"Stream Event => {stream_event.model_dump_json(indent=2, exclude_none=True)}")
                        # Track final state from streaming events
                        if stream_event.result and isinstance(stream_event.result, TaskStatusUpdateEvent):
                             if stream_event.result.final:
                                  final_state = stream_event.result.status.state
                    click.echo("--- End of Stream ---")
                    # After streaming, get the final task state if needed (e.g., for history)
                    if history or not final_state: # Get if history needed or final state wasn't received
                         get_task_resp = await client.get_task({"id": task_id, "historyLength": 10 if history else 0})
                         if get_task_resp.result:
                              task_result = get_task_resp.result
                              final_state = task_result.status.state
                         elif get_task_resp.error:
                              click.echo(f"Error getting final task state: {get_task_resp.error}", err=True)

                else: # Non-streaming
                    click.echo("--- Sending Request (Non-Streaming) ---")
                    send_task_resp = await client.send_task(payload)
                    click.echo(f"Response => {send_task_resp.model_dump_json(indent=2, exclude_none=True)}")
                    if send_task_resp.result:
                        task_result = send_task_resp.result
                        final_state = task_result.status.state
                        # For non-streaming, need to poll if state is WORKING or SUBMITTED
                        while final_state in [TaskState.WORKING, TaskState.SUBMITTED]:
                             click.echo(f"Task state is {final_state}. Polling again in 2 seconds...")
                             await asyncio.sleep(2)
                             get_task_resp = await client.get_task({"id": task_id, "historyLength": 0})
                             if get_task_resp.result:
                                  task_result = get_task_resp.result
                                  final_state = task_result.status.state
                                  click.echo(f"Polled State => {final_state}")
                             else:
                                  click.echo(f"Error polling task state: {get_task_resp.error}", err=True)
                                  break # Stop polling on error
                    elif send_task_resp.error:
                         click.echo(f"Error sending task: {send_task_resp.error}", err=True)


                # --- Handle Final State ---
                if final_state == TaskState.INPUT_REQUIRED:
                    click.echo("Agent requires more input for this task.")
                    # Loop continues naturally for the same task ID? No, spec implies new task ID usually.
                    # Let's stick to new task ID per loop iteration for simplicity.
                    # If INPUT_REQUIRED should continue the *same* task, logic needs adjustment here.
                    click.echo("Starting new interaction...")

                elif final_state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                    click.echo(f"Task finished with state: {final_state}")
                    if history and task_result:
                        click.echo("========= Task History =========")
                        # Ensure history is present and format it
                        history_list = task_result.history if task_result.history else []
                        for i, msg in enumerate(history_list):
                             click.echo(f"--- Message {i+1} ({msg.role}) ---")
                             for part in msg.parts:
                                  if isinstance(part, TextPart): click.echo(f"Text: {part.text}")
                                  elif isinstance(part, FilePart): click.echo(f"File: {part.file.name or '[no name]'}")
                                  elif isinstance(part, DataPart): click.echo(f"Data: {json.dumps(part.data, indent=2)}")
                             click.echo("-" * 20)
                        click.echo("==============================")

                else: # Unknown or still working after polling/streaming ended?
                     click.echo(f"Task ended with unexpected or intermediate state: {final_state}. Starting new interaction.")


            except Exception as e:
                logger.exception(f"An error occurred during task interaction (Task ID: {task_id}).")
                click.echo(f"Error during task: {e}", err=True)
                # Decide whether to break or continue the loop on error
                click.echo("Attempting to continue...")


    finally:
        # --- Cleanup ---
        if listener:
            click.echo("Stopping push notification listener...")
            listener.stop() # Request server shutdown
        click.echo("Exiting CLI.")


if __name__ == "__main__":
    asyncio.run(cli())
```

---
