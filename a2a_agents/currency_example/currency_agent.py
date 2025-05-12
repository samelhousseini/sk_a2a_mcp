import requests
import asyncio
import uuid
from datetime import datetime
import time
import sys
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# Adjust path to import from the common module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'python')))

# Import the helper functions from a2a_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    extract_text_from_part,
    extract_data_from_part
)

from common.types import (
    AgentCard, AgentSkill, AgentCapabilities, AgentAuthentication, AgentProvider,
    Task, TaskState, TaskStatus, Message, TextPart, DataPart, Part,
    Artifact, TaskSendParams, SendTaskRequest, SendTaskResponse, 
    SendTaskStreamingRequest, SendTaskStreamingResponse, 
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent,
)
from common.server import A2AServer, InMemoryTaskManager

# Basic configuration
AGENT_BASE_URL = "http://localhost:8001" 
FRANKFURTER_API_BASE = "https://api.frankfurter.dev/v1"

# Define Agent Skill and Agent Card
currency_skill = AgentSkill(
    id="currency_exchange_a2a_v3",
    name="Currency Exchange (A2AServer)",
    description="Converts an amount from one currency to another using official A2A types and A2AServer.",
    examples=["Convert 100 USD to EUR via A2AServer"],
    inputModes=["application/json"],
    outputModes=["application/json", "text/plain"]
)

agent_card_model = AgentCard(
    name="Currency Converter Agent (A2AServer)",
    description="Converts currencies using Frankfurter API with SSE and standard communication.",
    url=AGENT_BASE_URL,
    version="1.1.0",
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

class MyCurrencyAgentTaskManager(InMemoryTaskManager):
    def __init__(self):
        super().__init__()

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles non-streaming currency conversion requests using enhanced helper functions"""
        await self.upsert_task(request.params)
        
        task_id = request.params.id
        
        try:
            # Extract input parameters using the helper functions
            if not request.params.message or not request.params.message.parts:
                raise ValueError("Expected message with parts")
                
            # Extract data from the first part that contains data
            data_content = None
            for part in request.params.message.parts:
                data_content = extract_data_from_part(part)
                if data_content:
                    break
                    
            if not data_content:
                raise ValueError("Expected DataPart with amount, from_currency, to_currency")
                
            amount = data_content.get("amount")
            from_currency = data_content.get("from_currency")
            to_currency = data_content.get("to_currency")

            if not all([isinstance(amount, (int, float)), isinstance(from_currency, str), isinstance(to_currency, str)]):
                raise ValueError("Invalid or missing parameters: amount, from_currency, to_currency")

            # Update task status to working using the helper function with text_message
            await update_task_status(
                self, task_id, TaskState.WORKING, 
                text_message=f"Converting {amount} {from_currency} to {to_currency}."
            )

            # Get exchange rate
            response = requests.get(f"{FRANKFURTER_API_BASE}/latest?base={from_currency.upper()}&symbols={to_currency.upper()}")
            response.raise_for_status()
            rate_data = response.json()
            
            if not rate_data.get("rates") or to_currency.upper() not in rate_data["rates"]:
                raise ValueError(f"Could not find exchange rate for {to_currency.upper()}")
            
            exchange_rate = rate_data["rates"][to_currency.upper()]
            converted_amount = amount * exchange_rate

            # Add result as artifact using the helper function with both text_content and data_content
            result_text = f"{amount} {from_currency} is equal to {converted_amount:.2f} {to_currency}."
            
            await add_task_artifact(
                self, task_id,
                name="conversion_result",
                data_content={
                    "original_amount": amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "exchange_rate": exchange_rate,
                    "converted_amount": converted_amount
                },
                text_content=result_text
            )
            
            # Update task status to completed
            task = await update_task_status(self, task_id, TaskState.COMPLETED, text_message=result_text)
            
            # Return the response
            return SendTaskResponse(id=request.id, result=task)
            
        except ValueError as ve:
            task = await update_task_status(self, task_id, TaskState.FAILED, text_message=f"Error: {str(ve)}")
            return SendTaskResponse(id=request.id, result=task)
        except Exception as e:
            task = await update_task_status(self, task_id, TaskState.FAILED, text_message=f"Unexpected error: {str(e)}")
            return SendTaskResponse(id=request.id, result=task)

    async def _process_currency_conversion(self, request: SendTaskStreamingRequest):
        """Helper function for streaming currency conversion using enhanced helper functions"""
        params: TaskSendParams = request.params
        task_id = params.id
        
        try:
            # Send working status using text_message
            await send_status_update_event(
                self, task_id, TaskState.WORKING, 
                text_message="Processing currency exchange request."
            )

            # Extract input parameters using the helper functions
            if not params.message or not params.message.parts:
                raise ValueError("Expected message with parts")
                
            # Extract data from the first part that contains data
            data_content = None
            for part in params.message.parts:
                data_content = extract_data_from_part(part)
                if data_content:
                    break
                    
            if not data_content:
                raise ValueError("Expected DataPart with amount, from_currency, to_currency")
                
            amount = data_content.get("amount")
            from_currency = data_content.get("from_currency")
            to_currency = data_content.get("to_currency")

            if not all([isinstance(amount, (int, float)), isinstance(from_currency, str), isinstance(to_currency, str)]):
                raise ValueError("Invalid or missing parameters: amount, from_currency, to_currency")

            await send_status_update_event(
                self, task_id, TaskState.WORKING, 
                text_message=f"Converting {amount} {from_currency} to {to_currency}."
            )

            # Get available currencies
            try:
                response = await asyncio.to_thread(requests.get, f"{FRANKFURTER_API_BASE}/currencies")
                response.raise_for_status()
                available_currencies = response.json()
                
                # Use the helper function with data_content
                await add_task_artifact_event(
                    self, task_id, 
                    name="available_currencies_info", 
                    data_content=available_currencies, 
                    final_artifact_chunk=False
                )
            except Exception as e:
                await send_status_update_event(
                    self, task_id, TaskState.FAILED, 
                    text_message=f"Error fetching currencies: {str(e)}", 
                    final=True
                )
                return

            # Validate currencies
            if from_currency.upper() not in available_currencies or to_currency.upper() not in available_currencies:
                await send_status_update_event(
                    self, task_id, TaskState.FAILED, 
                    text_message="One or more currencies not supported", 
                    final=True
                )
                return
              # Get exchange rate
            try:
                # Add a streaming update to show we're getting the exchange rate
                await send_status_update_event(
                    self, task_id, TaskState.WORKING, 
                    text_message=f"Fetching current exchange rate for {from_currency} to {to_currency}..."
                )
                
                rate_url = f"{FRANKFURTER_API_BASE}/latest?base={from_currency.upper()}&symbols={to_currency.upper()}"
                rate_response = await asyncio.to_thread(requests.get, rate_url)
                rate_response.raise_for_status()
                rate_data = rate_response.json()
                
                # Another streaming update showing we got the rate
                await send_status_update_event(
                    self, task_id, TaskState.WORKING, 
                    text_message=f"Exchange rate retrieved successfully!"
                )
                
                if not rate_data.get("rates") or to_currency.upper() not in rate_data["rates"]:
                    raise ValueError(f"Could not find exchange rate for {to_currency.upper()}")
                
                exchange_rate = rate_data["rates"][to_currency.upper()]
                converted_amount = amount * exchange_rate
            except Exception as e:
                await send_status_update_event(
                    self, task_id, TaskState.FAILED, 
                    text_message=f"Error with exchange rate: {str(e)}", 
                    final=True
                )
                return            # Send final result using the updated helper functions with both data_content and text_content
            result_text = f"{amount} {from_currency} is equal to {converted_amount:.2f} {to_currency}."            # Send some streaming updates showing the calculation steps
            await send_status_update_event(
                self, task_id, TaskState.WORKING, 
                text_message=f"Calculating conversion using rate: {exchange_rate}"
            )
            
            # Remove sleep for more responsive streaming
            # Sending multiple status updates to demonstrate streaming
            await send_status_update_event(
                self, task_id, TaskState.WORKING,
                text_message=f"Converting {amount} {from_currency} to {to_currency}..."
            )
            
            await add_task_artifact_event(
                self, task_id,
                name="conversion_result",
                data_content={
                    "original_amount": amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "exchange_rate": exchange_rate,
                    "converted_amount": converted_amount
                },
                text_content=result_text,                
                final_artifact_chunk=True
            )

            # Send final status update immediately without sleep
            await send_status_update_event(
                self, task_id, TaskState.COMPLETED, 
                text_message=result_text, 
                final=True
            )

        except Exception as e:
            await send_status_update_event(
                self, task_id, TaskState.FAILED, 
                text_message=f"Error: {str(e)}", 
                final=True
            )

    async def on_send_task_subscribe(self, request: SendTaskStreamingRequest) -> AsyncGenerator[SendTaskStreamingResponse, None]:
        """Handles streaming tasks using SSE"""
        params: TaskSendParams = request.params
        task_id = params.id if params.id else str(uuid.uuid4())
        params.id = task_id
        
        is_new_task_session = True
        await self.upsert_task(params)
        sse_event_queue = await self.setup_sse_consumer(task_id=task_id)

        if is_new_task_session:
            asyncio.create_task(self._process_currency_conversion(request))

        return self.dequeue_events_for_sse(
            request_id=request.id,
            task_id=task_id,
            sse_event_queue=sse_event_queue,
        )

# Create and Start the A2A server
task_manager = MyCurrencyAgentTaskManager()
server = A2AServer(host="0.0.0.0", port=8001, 
                   agent_card=agent_card_model, 
                   task_manager=task_manager)

if __name__ == "__main__":
    print(f"Starting Currency Converter Agent at http://0.0.0.0:8001")
    print(f"Agent Card will be served at: http://0.0.0.0:8001/.well-known/agent.json")
    print(f"Endpoints: POST /tasks/send (standard), POST /tasks/sendSubscribe (SSE)")
    server.start()
