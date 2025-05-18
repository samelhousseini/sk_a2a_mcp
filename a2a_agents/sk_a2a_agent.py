'''
SK_A2A_Agent: A Semantic Kernel agent that integrates A2A communication capabilities.
'''
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterable, Union, Callable, Tuple, ClassVar
from common.server import A2AServer

# For InMemoryTaskManager and its types
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    SendTaskRequest,
    SendTaskResponse,
    TaskSendParams,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    JSONRPCResponse,
    AgentCard, 
    TaskState,
    Message
)

# For ChatCompletionAgent and its types
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.functions.kernel_plugin import KernelPlugin
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent

from semantic_kernel.agents import (
    ChatHistoryAgentThread,
)


import logging
logger = logging.getLogger(__name__)

from rich.console import Console
console = Console()

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

# For BaseA2AClient
from a2a_agents.base_a2a_client import BaseA2AClient

logger = logging.getLogger(__name__)

from semantic_kernel.filters import FilterTypes
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.functions import kernel_function, FunctionResult
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent





class SK_A2A_Agent(InMemoryTaskManager):
    '''
    SK_A2A_Agent combines the capabilities of a Semantic Kernel ChatCompletionAgent
    with an InMemoryTaskManager for handling A2A (Agent-to-Agent) protocol tasks,
    and includes a BaseA2AClient for communication.
    '''

    def __init__(
        self,
        # Args for ChatCompletionAgent (all keyword-only in parent)
        kernel: Optional[Kernel] = None,  # Added kernel parameter
        chat_service: Optional[ChatCompletionClientBase] = None,
        agent_id: Optional[str] = None,  # maps to 'id' in ChatCompletionAgent
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = "You are a helpful AI assistant.",
        plugins: Optional[Union[List[KernelPlugin], Dict[str, KernelPlugin]]] = None,
        prompt_template_config: Optional[PromptTemplateConfig] = None,
        function_choice_behavior: Optional[FunctionChoiceBehavior] = None,
        agent_arguments: Optional[KernelArguments] = None,  # maps to 'arguments' in ChatCompletionAgent

        # Args for BaseA2AClient
        a2a_agent_card: Optional[AgentCard] = None,
        a2a_url: Optional[str] = None,
        a2a_timeout: float = 60.0,
        a2a_output_dir: str = "output",
        a2a_debug: bool = False,
    ):
        '''
        Initializes the SK_A2A_Agent.

        Args:
            kernel: The Semantic Kernel instance.
            chat_service: The chat completion service to use.
            agent_id: The unique ID for the agent.
            name: The name of the agent.
            description: A description of the agent.
            instructions: Instructions for the agent's behavior.
            plugins: Plugins to be used by the agent.
            prompt_template_config: Prompt template configuration.
            function_choice_behavior: Function choice behavior for the agent.
            agent_arguments: Kernel arguments for the agent.
            a2a_agent_card: Agent card for the BaseA2AClient.
            a2a_url: URL for the BaseA2AClient.
            a2a_timeout: Timeout for the BaseA2AClient.
            a2a_output_dir: Output directory for the BaseA2AClient.
            a2a_debug: Debug flag for the BaseA2AClient.
        '''
        # Initialize InMemoryTaskManager (takes no args other than self)
        super().__init__()

        if kernel is None:
            kernel = Kernel()

        @kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
        async def log_function_calls(
            context: AutoFunctionInvocationContext, next
        ):
            # Log request
            print(f"[Filter] Calling {context.function.plugin_name}-{context.function.name} with args {context.arguments}")
            await next(context)
            # Log response
            result = context.function_result.value
            print(f"[Filter] Result from {context.function.plugin_name}-{context.function.name}: {result}")


        # Initialize ChatCompletionAgent first (all args must be passed as keywords)
        if chat_service is None:
            # Default to AzureChatCompletion, assuming environment variables are set
            # This might need to be more robust or allow easier configuration
            try:
                chat_service = AzureChatCompletion()
                logger.info(f"SK_A2A_Agent {name or agent_id}: Defaulted to AzureChatCompletion.")
            except Exception as e:
                logger.warning(f"SK_A2A_Agent {name or agent_id}: Failed to default to AzureChatCompletion: {e}. Chat service remains None. SK Agent might not function fully.")
                chat_service = None # Ensure it's None if default fails

        print("Plugins:", plugins)
        self.agent = ChatCompletionAgent(
            # kernel=kernel,
            service=chat_service,
            id=agent_id,
            name=name,
            description=description,
            instructions=instructions,
            plugins=plugins, 
            prompt_template_config=prompt_template_config,
            function_choice_behavior=function_choice_behavior,
            arguments=agent_arguments
        )
        
        self.a2a_client = None # Initialize to None
        if a2a_url or a2a_agent_card: # Only initialize if URL or card is provided
            try:
                self.a2a_client = BaseA2AClient(
                    agent_card=a2a_agent_card,
                    url=a2a_url,
                    timeout=a2a_timeout,
                    output_dir=a2a_output_dir,
                    debug=a2a_debug,
                    logger_instance=logger # Pass logger
                )
                logger.info(f"SK_A2A_Agent {name or agent_id}: BaseA2AClient initialized for URL '{a2a_url}' or provided card.")
            except ValueError as e:
                logger.error(f"SK_A2A_Agent {name or agent_id}: Failed to initialize BaseA2AClient even with URL/card: {e}")
        else:
            logger.info(f"SK_A2A_Agent {name or agent_id}: BaseA2AClient not initialized as no a2a_url or a2a_agent_card was provided.")

    # Abstract methods from InMemoryTaskManager (which inherits from TaskManager)
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        '''Handles processing a task.'''

        await self.upsert_task(request.params)
        task_id = request.params.id
        
        # Extract input parameters using the helper functions
        if not request.params.message or not request.params.message.parts:
            raise ValueError("Expected message with parts")
        

        try:
            logger.info(f"SK_A2A_Agent received on_send_task: {request.params.message.parts[0].text}")

            # Extracting text and data parts
            extracted_parts = await self.extract_message_parts(task_id, request.params.message)
            agent_query = await self.process_extracted_parts(task_id, extracted_parts)

            # Call the agent to formulate a response
            response = await self.formulate_response(task_id, agent_query)
            console.print(f"Processing response for task {task_id}: {response}")

            response_dict = parse_message_parts(response)
            
            await add_task_artifact(
                self, task_id,
                name="processing_result",
                data_content={
                    "response": str(response_dict["data_parts"]),
                },
                text_content=str(response_dict["text_parts"]),
            )

            console.print(f"Added task artifact for task {task_id}: {response_dict}")        
            logger.info(f"SK_A2A_Agent::Added task artifact for task {task_id}: {response_dict}")    

            # After processing, you might want to update the task status
            task = await update_task_status(
                self, task_id, TaskState.COMPLETED, 
                text_message=f"Completed processing for task {task_id}:\n{str(response_dict)}"
            )

            console.print(f"Task {task_id} completed with result: {task}")
            logger.info(f"SK_A2A_Agent::Task {task_id} completed with result: {task}")

            # Return the response
            return SendTaskResponse(id=request.id, result=task)

        except ValueError as ve:
            task = await update_task_status(self, task_id, TaskState.FAILED, text_message=f"Error: {str(ve)}")
            return SendTaskResponse(id=request.id, result=task)
        except Exception as e:
            task = await update_task_status(self, task_id, TaskState.FAILED, text_message=f"Unexpected error: {str(e)}")
            return SendTaskResponse(id=request.id, result=task)
        
        
        
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        '''Handles subscribing to task updates.'''

        task_id = request.params.id

        task = self.tasks.get(task_id)
        if task is None:
            is_new_task_session = True
        else:
            is_new_task_session = False

        await self.upsert_task(request.params)
        
        # Extract input parameters using the helper functions
        if not request.params.message or not request.params.message.parts:
            raise ValueError("Expected message with parts")
        
        sse_event_queue = await self.setup_sse_consumer(task_id=task_id)

        if is_new_task_session:
            asyncio.create_task(self._process_sse_request(request))
        else:
            # If the task already exists, we can just send the status update
            if (task.status.state == TaskState.WORKING):
                await send_status_update_event(
                    self, task_id, task.status.state, 
                    text_message="The task is already in progress."                
                )
            else:
                await send_status_update_event(
                    self, task_id, task.status.state, 
                    text_message=f"The task state is {task.status.state}.",
                    final=True               
                )

        return self.dequeue_events_for_sse(
            request_id=request.id,
            task_id=task_id,
            sse_event_queue=sse_event_queue,
        )
    

    async def _process_sse_request(self, request: SendTaskStreamingRequest):
        params: TaskSendParams = request.params
        task_id = params.id
        
        try:
            # Send working status using text_message
            await send_status_update_event(
                self, task_id, TaskState.WORKING, 
                text_message="Started processing Task request."
            )

            extracted_parts = await self.extract_message_parts(task_id, request.params.message)
            agent_query = await self.process_extracted_parts(task_id, extracted_parts)

            await send_status_update_event(
                self, task_id, TaskState.WORKING, 
                text_message="Extracted and processed message parts."
            )

            # Call the agent to formulate a response
            response = await self.formulate_response(task_id, agent_query)

            console.print(f"Processing SSE response for task {task_id}: {response}")

            await send_status_update_event(
                self, task_id, TaskState.WORKING, 
                text_message="Collected Agent response."
            )

            text_contents = "\n".join(extract_all_text_parts(response))
            data_contents = extract_all_data_parts(response)
            file_contents = extract_all_file_parts(response)

            await add_task_artifact_event(
                self, task_id,
                name="processing_result",
                data_content={
                    "response": data_contents,
                },
                text_content=text_contents,
                file_part=file_contents,
                final_artifact_chunk=True
            )

            console.print(f"Task {task_id} completed with result: {text_contents}")
            logger.info(f"SK_A2A_Agent::Task {task_id} completed with result: {text_contents}")
            
            # Send final status update immediately without sleep
            await send_status_update_event(
                self, task_id, TaskState.COMPLETED,
                text_message=text_contents,
                data_content={
                    "response": data_contents,
                },
                file_part=file_contents,
                final=True
            )


        except Exception as e:
            await send_status_update_event(
                self, task_id, TaskState.FAILED, 
                text_message=f"Error: {str(e)}", 
                final=True
            )



    async def extract_message_parts(self, 
                             task_id: str,
                             message: Message) -> None:
        '''Handles incoming messages.'''            
        
        await update_task_status(
                self, task_id, TaskState.WORKING, 
                text_message=f"Extracting data from message for task {task_id}"
            )
        
        # Extract data from the first part that contains data
        data_contents = extract_all_data_parts(message)
        file_contents = extract_all_file_parts(message)
        text_contents = extract_all_text_parts(message)

        return {
            "data_contents": data_contents,
            "file_contents": file_contents,
            "text_contents": text_contents
        }
    

    async def process_extracted_parts(
        self,
        task_id: str,
        extracted_parts: Dict[str, Any]
    ) -> None:
        '''Processes the extracted parts.'''
        console.print(f"Processing extracted parts for task {task_id}")

        # Process the extracted parts as needed
        # For example, you might want to send them to another service or perform some computation
        data_contents = extracted_parts.get("data_contents", [])
        file_contents = extracted_parts.get("file_contents", [])
        text_contents = extracted_parts.get("text_contents", [])

        agent_query = {
            "task_id": task_id,
            "data_contents": data_contents,
            "file_contents": file_contents,
            "text_contents": text_contents
        }

        return agent_query


    # This method is to be overridden in subclasses
    @abstractmethod
    async def formulate_response(self, task_id: str, agent_query: Dict[str, Any]) -> Any:
        '''Formulates a response based on the agent query.'''
        
        thread = await self.get_chat_history_thread(task_id)

        console.print(f"Sending agent query for task {task_id}: {agent_query}")
        response = await self.agent.get_response(message=agent_query, thread=thread)
        console.print(f"Received response for task {task_id}: {response.content}")

        return response


    async def get_chat_history_thread(
        self, task_id: str
    ) -> Optional[ChatHistoryAgentThread]:
        '''Retrieves the chat history thread for a given task ID.'''
        thread = ChatHistoryAgentThread()
        task = self.tasks.get(task_id)

        if task:
            for message in task.history:
                data_contents = extract_all_data_parts(message)
                file_contents = extract_all_file_parts(message)
                text_contents = extract_all_text_parts(message)

                if message.role == "user":
                    # Append the message to the thread
                    for text in text_contents:
                        thread._chat_history.add_user_message(text)
                    
                    for data in data_contents:
                        thread._chat_history.add_user_message(data)

                    for file in file_contents:
                        thread._chat_history.add_user_message(file)
                    
                elif message.role == "agent":
                    # Append the message to the thread
                    for text in text_contents:
                        thread._chat_history.add_assistant_message(text)
                    
                    for data in data_contents:
                        thread._chat_history.add_assistant_message(data)

                    for file in file_contents:
                        thread._chat_history.add_assistant_message(file)
        
        console.print(f"Chat history thread for task {task_id}: {thread._chat_history}")

        return thread
    

    @staticmethod
    def get_agent_card(host: str, port: int) -> AgentCard:
        return None

    # This method starts the A2A server
    # and sets up the agent card and task manager.
    def start(self, host: str, port: int, agent_card: AgentCard = None):
        if agent_card is not None: self.agent_card = agent_card
        else: self.agent_card = self.get_agent_card(host, port)

        self.server = A2AServer(
            agent_card=self.agent_card,
            task_manager=self,
            host=host,
            port=port,
        )
        self.server.start()