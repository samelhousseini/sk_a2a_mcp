import sys
import os
import uuid

# Adjust path to import from the common module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the A2ACardServer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SSE.a2a_card_server import A2ACardServer

from common.types import (
    AgentCard, AgentSkill, AgentCapabilities, AgentAuthentication, AgentProvider,
    Task, TaskState, TaskStatus, Message, TextPart, DataPart, Part,
    Artifact, TaskSendParams, SendTaskRequest, SendTaskResponse, 
    SendTaskStreamingRequest, SendTaskStreamingResponse, 
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent,
)
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from common.server import A2AServer, InMemoryTaskManager

# Basic configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8001
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Define multiple agent cards
# 1. Currency Converter Agent
currency_skill = AgentSkill(
    id="currency_exchange_a2a_v3",
    name="Currency Exchange",
    description="Converts an amount from one currency to another.",
    examples=["Convert 100 USD to EUR"],
    inputModes=["application/json"],
    outputModes=["application/json", "text/plain"]
)

currency_agent_card = AgentCard(
    name="Currency Converter Agent",
    description="Converts currencies using Frankfurter API.",
    url=f"{BASE_URL}",
    version="1.0.0",
    provider=AgentProvider(name="A2A Example Corp", url="https://example.com/a2a", organization="Microsoft"),
    capabilities=AgentCapabilities(
        streaming=True,
        pushNotifications=False,
        stateTransitionHistory=True        
    ),
    authentication=AgentAuthentication(schemes=["no-auth"]),
    defaultInputModes=["application/json"],
    defaultOutputModes=["application/json", "text/plain"],
    skills=[currency_skill]
)

# 2. Weather Agent
weather_skill = AgentSkill(
    id="weather_forecast_a2a_v1",
    name="Weather Forecast",
    description="Provides weather forecasts for a location.",
    examples=["Get weather for Seattle"],
    inputModes=["application/json"],
    outputModes=["application/json", "text/plain"]
)

weather_agent_card = AgentCard(
    name="Weather Forecast Agent",
    description="Provides weather information using OpenWeather API.",
    url=f"{BASE_URL}",
    version="1.0.0",
    provider=AgentProvider(name="A2A Example Corp", url="https://example.com/a2a", organization="Microsoft"),
    capabilities=AgentCapabilities(
        streaming=True,
        pushNotifications=False,
        stateTransitionHistory=True        
    ),
    authentication=AgentAuthentication(schemes=["no-auth"]),
    defaultInputModes=["application/json"],
    defaultOutputModes=["application/json", "text/plain"],
    skills=[weather_skill]
)

# 3. Translation Agent
translation_skill = AgentSkill(
    id="translation_a2a_v1",
    name="Text Translation",
    description="Translates text between languages.",
    examples=["Translate 'Hello world' to Spanish"],
    inputModes=["application/json"],
    outputModes=["application/json", "text/plain"]
)

translation_agent_card = AgentCard(
    name="Translation Agent",
    description="Translates text between different languages.",
    url=f"{BASE_URL}",
    version="1.0.0",
    provider=AgentProvider(name="A2A Example Corp", url="https://example.com/a2a", organization="Microsoft"),
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=True        
    ),
    authentication=AgentAuthentication(schemes=["no-auth"]),
    defaultInputModes=["application/json"],
    defaultOutputModes=["application/json", "text/plain"],
    skills=[translation_skill]
)

# Define a simple task manager that routes tasks based on skill ID
class MultiAgentTaskManager(InMemoryTaskManager):
    def __init__(self):
        super().__init__()
        
    async def on_send_task_subscribe(self, request: SendTaskStreamingRequest) -> AsyncGenerator[SendTaskStreamingResponse, None]:
        """Handles streaming tasks using SSE"""
        pass
    
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Route tasks to the appropriate handler based on the skill ID"""
        await self.upsert_task(request.params)
        
        task_id = request.params.id
        
        try:
            # Extract skill ID from metadata if provided
            skill_id = None
            if request.params.metadata and "skill_id" in request.params.metadata:
                skill_id = request.params.metadata["skill_id"]
                
            # Route to the appropriate handler based on skill ID
            if skill_id == "currency_exchange_a2a_v3":
                return await self.handle_currency_exchange(request)
            elif skill_id == "weather_forecast_a2a_v1":
                return await self.handle_weather_forecast(request)
            elif skill_id == "translation_a2a_v1":
                return await self.handle_translation(request)
            else:
                # Default to a simple echo response
                message_text = "No specific handler for this request. Please specify a skill_id in metadata."
                task = await self._create_echo_response(task_id, request)
                return SendTaskResponse(id=request.id, result=task)
                
        except Exception as e:
            task = await self._update_task_status(task_id, TaskState.FAILED, f"Error: {str(e)}")
            return SendTaskResponse(id=request.id, result=task)
            
    async def _create_echo_response(self, task_id: str, request: SendTaskRequest) -> Task:
        """Create a simple echo response for demonstration purposes"""
        # Extract text if available
        message_text = "Echo: "
        if request.params.message and request.params.message.parts:
            for part in request.params.message.parts:
                if isinstance(part, TextPart):
                    message_text += part.text
                elif isinstance(part, DataPart):
                    message_text += f"Data: {part.data}"
        
        return await self._update_task_status(task_id, TaskState.COMPLETED, message_text)
        
    async def _update_task_status(self, task_id: str, state: TaskState, message_text: str) -> Task:
        """Helper to update task status"""
        agent_message = Message(role="agent", parts=[TextPart(type="text", text=message_text)])
        
        current_task = self.tasks[task_id]
        current_task.status.state = state
        current_task.status.message = agent_message
        
        if current_task.history is None:
            current_task.history = [agent_message]
        else:
            current_task.history.append(agent_message)
            
        return current_task
        
    async def handle_currency_exchange(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handle currency exchange requests"""
        task_id = request.params.id
        
        # Simple mock implementation
        task = await self._update_task_status(
            task_id, 
            TaskState.COMPLETED, 
            "This is a mock currency exchange response. In a real implementation, this would convert currencies."
        )
        return SendTaskResponse(id=request.id, result=task)
        
    async def handle_weather_forecast(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handle weather forecast requests"""
        task_id = request.params.id
        
        # Simple mock implementation
        task = await self._update_task_status(
            task_id, 
            TaskState.COMPLETED, 
            "This is a mock weather forecast response. In a real implementation, this would provide weather data."
        )
        return SendTaskResponse(id=request.id, result=task)
        
    async def handle_translation(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handle translation requests"""
        task_id = request.params.id
        
        # Simple mock implementation
        task = await self._update_task_status(
            task_id, 
            TaskState.COMPLETED, 
            "This is a mock translation response. In a real implementation, this would translate text."
        )
        return SendTaskResponse(id=request.id, result=task)

# Create and start the server
def main():
    # Create a list of agent cards
    agent_cards = [
        currency_agent_card,
        weather_agent_card,
        translation_agent_card
    ]
    
    # Create the task manager
    task_manager = MultiAgentTaskManager()
    
    # Create the A2ACardServer with multiple agent cards
    print(f"Starting Multi-Agent Server at http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"Agent Cards will be served at: http://{SERVER_HOST}:{SERVER_PORT}/.well-known/agents.json")
    print(f"Single Agent Card will be served at: http://{SERVER_HOST}:{SERVER_PORT}/.well-known/agent.json (returns the first card)")
    
    # Use the first card as the "default" single card for backward compatibility
    server = A2ACardServer(
        host=SERVER_HOST, 
        port=SERVER_PORT,
        agent_card=agent_cards[0],  # First card as the default
        agent_cards=agent_cards,    # All cards
        task_manager=task_manager
    )
    
    # Start the server
    server.start()

if __name__ == "__main__":
    main()
