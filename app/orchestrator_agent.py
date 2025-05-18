import os
import sys
import logging
import argparse
import json

sys.path.append("./")
sys.path.append("../")

from typing import Optional, Dict, Any, List
from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions.kernel_plugin import KernelPlugin
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.agents import ChatHistoryAgentThread

from a2a_agents.sk_a2a_agent import SK_A2A_Agent
from a2a_agents.base_a2a_client import BaseA2AClient
from common.types import Message, Part, TextPart, DataPart, TaskSendParams, TaskState
from common.types import AgentCard, AgentSkill, AgentCapabilities
from a2a_agents.a2a_utils.a2a_helper_functions import (
    extract_user_query, create_agent_response, extract_message_content,
    send_status_update_event, create_client_message, create_task_params,
    extract_text_from_part
)

from rich.console import Console
console = Console()

# Environment variables for helper agent URLs
FAQ_AGENT_URL = os.environ.get("FAQ_AGENT_URL", "http://localhost:8001")
TECH_AGENT_URL = os.environ.get("TECH_AGENT_URL", "http://localhost:8002")
ESCALATION_AGENT_URL = os.environ.get("ESCALATION_AGENT_URL", "http://localhost:8003")

# --- Support Router Plugin ---
class SupportRouterPlugin():
    """A Semantic Kernel plugin to route support requests to specialized agents."""
    faq_client: BaseA2AClient = None
    tech_client: BaseA2AClient = None
    escalation_client: BaseA2AClient = None
    logger: logging.Logger = None
    task_manager: SK_A2A_Agent = None
    
    async def _call_agent_internal(self, client: BaseA2AClient, task_id: str, user_query: str, data_payload: Optional[Dict[str, Any]] = None) -> Message:
        """Internal helper to call a specific agent and stream updates using SSE."""
        await send_status_update_event(self.task_manager, task_id, TaskState.WORKING, text_message=f"Routing to {client.url}")
        
        # Create task params
        message = create_client_message(text_content=user_query, data_content=data_payload)
        task_params = create_task_params(message, task_id=f"{task_id}_sub", session_id=task_id)
        
        # Call the agent with streaming
        _, stream = await client.send_task_streaming(task_params)
        
        # Process final response
        final_response = None
        async for event in stream:
            result = event.result
            if hasattr(result, 'status') and result.status.state == TaskState.COMPLETED and result.final:
                final_response = result.status.message
            elif hasattr(result, 'status') and result.status.state == TaskState.FAILED:
                final_response = Message(parts=[TextPart(text_content="An error occurred while processing your request.")])
        
        console.print(f"[blue]Final response from {client.url}:[/] {final_response}")
        return final_response

    @kernel_function(name="ask_faq_agent", description="Routes a user's question to the FAQ agent to get answers for frequently asked questions.")
    async def ask_faq_agent(self, user_query: str, task_id: str) -> str:
        if not self.faq_client:
            self.faq_client = BaseA2AClient(url=FAQ_AGENT_URL)
        
        response = await self._call_agent_internal(self.faq_client, task_id, user_query)
        return extract_text_from_part(response.parts[0]) if response and response.parts else "No answer from FAQ agent."

    @kernel_function(name="diagnose_technical_issue", description="Routes a user's technical problem to the Technical Troubleshooting agent to get diagnostic help.")
    async def diagnose_technical_issue(self, user_query: str, task_id: str, session_data_json: Optional[str] = None) -> str:
        if not self.tech_client:
            self.tech_client = BaseA2AClient(url=TECH_AGENT_URL)
        
        data_payload = json.loads(session_data_json) if session_data_json else None
        response = await self._call_agent_internal(self.tech_client, task_id, user_query, data_payload)
        return extract_text_from_part(response.parts[0]) if response and response.parts else "No diagnosis from technical agent."

    @kernel_function(name="escalate_to_human_support", description="Escalates the user's issue to human support. Gathers necessary case details.")
    async def escalate_to_human_support(self, user_query: str, task_id: str, case_details_json: Optional[str] = None) -> str:
        if not self.escalation_client:
            self.escalation_client = BaseA2AClient(url=ESCALATION_AGENT_URL)
        
        data_payload = json.loads(case_details_json) if case_details_json else None
        response = await self._call_agent_internal(self.escalation_client, task_id, user_query, data_payload)
        return extract_text_from_part(response.parts[0]) if response and response.parts else "No response from escalation agent."

# --- Orchestrator Agent ---
class OrchestratorAgent(SK_A2A_Agent):
    """Orchestrates customer support inquiries by routing them to specialized helper agents."""

    def __init__(
        self,
        agent_id: Optional[str] = "OrchestratorAgent",
        name: Optional[str] = "CustomerSupportOrchestrator",
        description: Optional[str] = "Orchestrates customer support requests between specialized agents using an LLM.",
        host: str = "localhost",
        port: int = 8000,
    ):
        # System prompt for the orchestrator's LLM
        instructions = """You are a customer support orchestrator for a technology company. Analyze customer queries and route them to the appropriate specialized agent:

1. For general product questions and information, use ask_faq_agent
2. For technical problems requiring troubleshooting, use diagnose_technical_issue
3. For issues requiring human intervention, use escalate_to_human_support

Always be helpful, professional, and empathetic. Inform the user which specialized service you're routing them to."""
                
        # Create the support router plugin
        self.router_plugin = SupportRouterPlugin()
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            instructions=instructions,
            plugins=[self.router_plugin]
        )
        
        # Store agent threads for different sessions
        self.agent_threads = {}
        
        # Set the router's task manager to self for status updates
        self.router_plugin.task_manager = self
        self.router_plugin.logger = logging.getLogger("SupportRouterPlugin")
        
        console.print(f"[bold green]Orchestrator Agent:[/] {self.agent.name} initialized with ID: {self.agent.id}")
    
    async def formulate_response(self, task_id: str, agent_query: Dict[str, Any]) -> Message:
        """Route user query to specialized agents based on content."""
        await send_status_update_event(self, task_id, TaskState.WORKING, text_message="Analyzing your request...")
        
        user_query, _ = extract_user_query(agent_query, log_prefix=f"[{self.agent.name} - {task_id}]", logger_instance=logging.getLogger())
        user_query = f"User Query: {user_query}\n\nTask ID: {task_id}\n\nPlease route this request to the appropriate specialized agent."

        console.print(f"[cyan]Orchestrator Agent:[/] Processing query: [yellow]{user_query[:50]}...[/]")
        
        # Create or get an agent thread for this session
        thread_id = f"thread_{task_id}"
        if thread_id not in self.agent_threads:
            self.agent_threads[thread_id] = ChatHistoryAgentThread()
        
        thread = self.agent_threads[thread_id]
        
        async for response in self.agent.invoke(messages=user_query, thread=thread):                
            self.agent_threads[thread_id] = response.thread                    
            content = response.content
        
        return create_agent_response(text_content=str(content))

    @staticmethod
    def get_agent_card(host: str, port: int) -> AgentCard:
        orchestration_skill = AgentSkill(
            id="orchestration-skill",
            name="CustomerSupportOrchestration",
            description="Understands customer queries and routes them to specialized agents using LLM-based reasoning.",
            tags=["orchestration", "routing", "customer support", "llm"],
            examples=["I need help with my internet.", "What are your business hours?", "I want to speak to a manager."],
            defaultInputModes=["text/plain", "application/json"],
            defaultOutputModes=["text/plain", "application/json"],
        )
        capabilities = AgentCapabilities(streaming=True)        
        return AgentCard(
            name="Customer Support Orchestrator (LLM-Powered)",
            description="Orchestrates customer support requests between specialized agents using LLM-based reasoning.",
            url=f"http://{host}:{port}/",
            version="0.3.0", # Updated version for SSE streaming implementation
            defaultInputModes=["text/plain", "application/json"],
            defaultOutputModes=["text/plain", "application/json"],
            capabilities=capabilities,
            skills=[orchestration_skill]
        )

# Ensure __main__ block is present and correct for starting the agent
if __name__ == "__main__":
    # Setup logging for the main script entry point
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run the Orchestrator Agent.")
    parser.add_argument("--host", type=str, default="localhost", help="Host for the agent server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the agent server.")
    args = parser.parse_args()    
    
    def main():
        logger.info(f"Starting Orchestrator Agent on {args.host}:{args.port}")
        agent = OrchestratorAgent(host=args.host, port=args.port)
        agent.start(host=args.host, port=args.port)
        logger.info("Orchestrator Agent server has been started.")

    # Call main directly - no need for asyncio.run since start() creates its own event loop
    main()

