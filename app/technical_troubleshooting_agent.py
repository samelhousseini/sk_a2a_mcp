import os
import sys
import logging
import argparse

sys.path.append("./")
sys.path.append("../")
from a2a_agents.sk_a2a_agent import SK_A2A_Agent
from common.types import AgentSkill, AgentCapabilities, AgentCard, TaskState, Message, TextPart, DataPart
from typing import Dict, Any, Optional, List
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

class TechnicalTroubleshootingAgent(SK_A2A_Agent):
    """Agent to diagnose technical issues based on user input."""    
    def __init__(
        self,
        agent_id: Optional[str] = "TechTroubleshootAgent",
        name: Optional[str] = "TechnicalTroubleshootingAgent",
        description: Optional[str] = "Diagnoses technical issues and suggests solutions.",
        instructions: Optional[str] = """You are a technical troubleshooting assistant. Guide users through diagnostic steps.

Use the following tools to help diagnose technical issues:
1. get_diagnostic_steps - Get a single step for troubleshooting an issue
2. get_all_diagnostic_steps - Get all steps for troubleshooting an issue
3. get_advanced_diagnostics - Get advanced information for technical experts

Available issue types: internet_connectivity, slow_computer, printer_issues, software_crashes, email_issues""",          
        mcp_url: Optional[str] = "http://localhost:8010/sse",
        **kwargs,
    ):
        self.mcp_url = mcp_url
        
        # Initialize with MCP client
        self.mcp_client = MCPSsePlugin(
            url=self.mcp_url,
            name="technical_troubleshooting",
            description="Technical Troubleshooting Plugin. For any technical diagnostics, please call this plugin."
        )
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            instructions=instructions,
            plugins=[self.mcp_client],
            **{k: v for k, v in kwargs.items() if k not in ['host', 'port']}
        )
        
        # Dictionary to store agent threads for persistent sessions
        self.agent_threads = {}
        console.print(f"[bold green]Technical Troubleshooting Agent:[/] {self.agent.name} initialized with ID: {self.agent.id}")
        
    async def formulate_response(self, task_id: str, agent_query: Dict[str, Any]) -> Message:
        """Formulates a response by guiding the user through diagnostic steps."""
        await send_status_update_event(self, task_id, TaskState.WORKING, text_message="Processing your technical troubleshooting request...")
        
        user_query, _ = extract_user_query(agent_query, log_prefix=f"[{self.agent.name} - {task_id}]", logger_instance=logger)
        console.print(f"[cyan]Technical Troubleshooting Agent:[/] Processing query: [yellow]{user_query[:50]}...[/]")
        
        # Create or get an agent thread for this task_id (session)
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
            id="tech-troubleshooting-skill",
            name="Technical Troubleshooting Skill",
            description="Helps diagnose and resolve common technical problems.",
            tags=["technical support", "troubleshooting", "diagnostics"],
            examples=["My internet is down", "My computer is very slow", "I can't print"],
            inputModes=["text/plain", "application/json"], # JSON for session state
            outputModes=["text/plain", "application/json"],
        )
        capabilities = AgentCapabilities(streaming=True)
        return AgentCard(            
            name="Technical Troubleshooting Agent",
            description="This agent guides users through technical troubleshooting steps.",
            url=f"http://{host}:{port}/",
            version="0.2.0", # Updated version for SSE streaming implementation
            defaultInputModes=["text/plain", "application/json"],
            defaultOutputModes=["text/plain", "application/json"],
            capabilities=capabilities,
            skills=[skill]
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run the Technical Troubleshooting Agent.")
    parser.add_argument("--host", type=str, default="localhost", help="Host for the agent server.")
    parser.add_argument("--port", type=int, default=8002, help="Port for the agent server.")
    args = parser.parse_args()

    logger.info(f"Starting Technical Troubleshooting Agent on {args.host}:{args.port}")
    agent = TechnicalTroubleshootingAgent()
    agent.start(host=args.host, port=args.port)
