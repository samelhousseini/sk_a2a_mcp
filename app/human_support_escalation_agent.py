import os
import sys
import logging
import argparse

sys.path.append("./")
sys.path.append("../")

from a2a_agents.sk_a2a_agent import SK_A2A_Agent
from common.types import AgentSkill, AgentCapabilities, AgentCard, TaskState, Message, TextPart, DataPart
from typing import Optional, Dict, Any, List
from a2a_agents.a2a_utils.a2a_helper_functions import (
    extract_user_query, create_agent_response, extract_message_content,
    send_status_update_event
)
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.functions.kernel_plugin import KernelPlugin
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin, MCPSsePlugin

from rich.console import Console
console = Console()

logger = logging.getLogger(__name__)

class HumanSupportEscalationAgent(SK_A2A_Agent):
    """Agent to escalate issues to human support and prepare case details."""

    def __init__(
        self,
        agent_id: Optional[str] = "HumanEscalationAgent",
        name: Optional[str] = "HumanSupportEscalationAgent",
        description: Optional[str] = "Escalates issues to human support and prepares case details.",
        instructions: Optional[str] = """You are a human support escalation assistant. Collect necessary details and confirm escalation. 

Use the following tools to manage support cases:
1. create_support_case - Create a new support case with customer details
2. get_case_status - Check the status of an existing case
3. update_support_case - Update information on an existing case

When collecting information, ask for the customer's name, contact information, and a detailed description of their issue. 
Assign an appropriate priority level (low, medium, high, critical) based on the issue severity.""",          
        mcp_url: Optional[str] = "http://localhost:8010/sse",
        **kwargs,
    ):
        self.mcp_url = mcp_url
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            instructions=instructions,
            plugins=[],
            **{k: v for k, v in kwargs.items() if k not in ['host', 'port']}
        )
        
        # Dictionary to store agent threads for persistent sessions
        self.agent_threads = {}
        console.print(f"[bold green]Human Support Escalation Agent:[/] {self.agent.name} initialized with ID: {self.agent.id}")
        
    async def formulate_response(self, task_id: str, agent_query: Dict[str, Any]) -> Message:
        """Formulates a response by confirming escalation and providing a case ID."""
        await send_status_update_event(self, task_id, TaskState.WORKING, text_message="Processing your escalation request...")
        
        user_query, _ = extract_user_query(agent_query, log_prefix=f"[{self.agent.name} - {task_id}]", logger_instance=logger)
        console.print(f"[cyan]Human Support Escalation Agent:[/] Processing query: [yellow]{user_query[:50]}...[/]")
        
        # Create or get an agent thread for this task_id (session)
        thread_id = f"thread_{task_id}" 
        if thread_id not in self.agent_threads:
            self.agent_threads[thread_id] = ChatHistoryAgentThread()
        
        thread = self.agent_threads[thread_id]
        
        # Use MCP client for case management
        async with MCPSsePlugin(url=self.mcp_url, name="support_case_manager") as mcp_client:
            self.agent.kernel.plugins = []
            self.agent.kernel.add_plugins([mcp_client])
            
            async for response in self.agent.invoke(messages=user_query, thread=thread):                
                self.agent_threads[thread_id] = response.thread                    
                content = response.content
        
        return create_agent_response(text_content=str(content))
        
    @staticmethod
    def get_agent_card(host: str, port: int) -> AgentCard:
        skill = AgentSkill(
            id="escalate-to-human-skill",
            name="EscalateToHuman",
            description="Collects information and escalates the support ticket to a human agent.",
            tags=["escalation", "human support", "case management"],
            examples=["I need to speak to a human agent about my billing issue.", "Please escalate my problem with product X."],
            defaultInputModes=["application/json"], # Expects structured data for escalation
            defaultOutputModes=["application/json"],
        )
        capabilities = AgentCapabilities(streaming=True)

        return AgentCard(
            name="Human Support Escalation Agent", # AgentCard name can have spaces
            description="Escalates issues to human support and prepares case details.",
            url=f"http://{host}:{port}/",
            version="0.2.0", # Incremented version to reflect streaming capabilities
            defaultInputModes=["application/json"], # Expects structured data for escalation
            defaultOutputModes=["application/json"],
            capabilities=capabilities,
            skills=[skill]
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Run the Human Support Escalation Agent.")
    parser.add_argument("--host", type=str, default="localhost", help="Host for the agent server.")
    parser.add_argument("--port", type=int, default=8003, help="Port for the agent server.")
    args = parser.parse_args()

    logger.info(f"Starting Human Support Escalation Agent on {args.host}:{args.port}")
    agent = HumanSupportEscalationAgent() 
    agent.start(host=args.host, port=args.port)
