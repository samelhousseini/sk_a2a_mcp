import os
import sys
import logging
import argparse

sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")
sys.path.append(".")
from a2a_agents.sk_a2a_agent import SK_A2A_Agent
from common.types import AgentSkill, AgentCapabilities, AgentCard, TaskState, Message, TextPart
from typing import Dict, Any, Optional
from a2a_agents.a2a_utils.a2a_helper_functions import (
    extract_user_query, create_agent_response, log_agent_response, extract_message_content,
    send_status_update_event
)
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.functions.kernel_plugin import KernelPlugin
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin, MCPSsePlugin
from rich.console import Console
console = Console()

logger = logging.getLogger(__name__)

class FAQAgent(SK_A2A_Agent):
    """Agent to answer frequently asked questions from a knowledge base."""

    def __init__(
        self,
        agent_id: Optional[str] = "FAQAgent",
        name: Optional[str] = "FAQAgent",
        description: Optional[str] = "Handles common questions by querying a knowledge base.",
        instructions: Optional[str] = "You are an FAQ assistant. Answer questions based on the provided knowledge base. Use the search_faq tool to find answers to user questions. If you're not sure which topics are available, use the get_faq_topics tool to see a list of topics.",          
        mcp_url: Optional[str] = "http://localhost:8010/sse",
        **kwargs,
    ):
        self.mcp_url = mcp_url
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            instructions=instructions,
            **{k: v for k, v in kwargs.items() if k not in ['host', 'port']},
        )
        
        self.agent_threads = {}
        console.print(f"[bold green]FAQ Agent:[/] {self.agent.name} initialized with ID: {self.agent.id}")
        
    async def formulate_response(self, task_id: str, agent_query: Dict[str, Any]) -> Message:
        """Formulates a response based on the agent query using Semantic Kernel capabilities."""
        await send_status_update_event(self, task_id, TaskState.WORKING, text_message="Processing your FAQ request...")
        
        user_query, _ = extract_user_query(agent_query, log_prefix=f"[{self.agent.name} - {task_id}]", logger_instance=logger)
        console.print(f"[cyan]FAQ Agent:[/] Processing query: [yellow]{user_query[:50]}...[/]")
        
        thread_id = f"thread_{task_id}"
        if thread_id not in self.agent_threads:
            self.agent_threads[thread_id] = ChatHistoryAgentThread()
        
        thread = self.agent_threads[thread_id]
        
        # Use an MCP plugin to query knowledge base
        async with MCPSsePlugin(url=self.mcp_url, name="faq_plugin") as mcp_client:
            self.agent.kernel.plugins = []
            self.agent.kernel.add_plugins([mcp_client])
            
            async for response in self.agent.invoke(messages=user_query, thread=thread):                
                self.agent_threads[thread_id] = response.thread                    
                content = response.content
        
        return create_agent_response(text_content=str(content))

    @staticmethod
    def get_agent_card(host: str, port: int) -> AgentCard:
        skill = AgentSkill(
            id="faq-skill",
            name="FAQ Skill",
            description="Answers frequently asked questions.",
            tags=["faq", "knowledge base"],
            examples=["What is your name?", "How to reset password?"],
            inputModes=["text/plain"],
            outputModes=["text/plain"],
        )
        capabilities = AgentCapabilities(streaming=True) # Assuming SK_A2A_Agent supports streaming        
        return AgentCard(
            name="FAQ Agent",
            description="This agent answers common questions from a knowledge base.",
            url=f"http://{host}:{port}/", # URL where this agent will be hosted
            version="0.2.0", # Updated version for SSE streaming implementation
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            capabilities=capabilities,
            skills=[skill]
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run the FAQ Agent.")
    parser.add_argument("--host", type=str, default="localhost", help="Host for the agent server.")
    parser.add_argument("--port", type=int, default=8001, help="Port for the agent server.")
    args = parser.parse_args()

    logger.info(f"Starting FAQ Agent on {args.host}:{args.port}")
    agent = FAQAgent()
    agent.start(host=args.host, port=args.port)
