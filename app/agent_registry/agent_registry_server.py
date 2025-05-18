#!/usr/bin/env python
"""
Multi-Agent Server - Starts a server hosting all 4 agents' cards together
"""
import os
import sys
import logging

import sys
sys.path.append("../")
sys.path.append("../../")

# Adjust path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from a2a_agents.a2a_card_server import A2ACardServer
from common.server import A2AServer, InMemoryTaskManager
from common.types import AgentCard

# Import all agent classes to use their static get_agent_card methods
from orchestrator_agent import OrchestratorAgent
from faq_agent import FAQAgent
from technical_troubleshooting_agent import TechnicalTroubleshootingAgent
from human_support_escalation_agent import HumanSupportEscalationAgent

from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from common.types import (
    SendTaskRequest, SendTaskResponse, 
    SendTaskStreamingRequest, SendTaskStreamingResponse, 
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration for the multi-agent server
SERVER_HOST = "localhost"
SERVER_PORT = 8888
AGENT_CONFIG = [
    {"type": "orchestrator", "class": OrchestratorAgent, "port": 8000},
    {"type": "faq", "class": FAQAgent, "port": 8001},
    {"type": "tech", "class": TechnicalTroubleshootingAgent, "port": 8002},
    {"type": "escalation", "class": HumanSupportEscalationAgent, "port": 8003}
]


class DummyTaskManager(InMemoryTaskManager):
    def __init__(self):
        super().__init__()

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles task sending"""
        pass

    async def on_send_task_subscribe(self, request: SendTaskStreamingRequest) -> AsyncGenerator[SendTaskStreamingResponse, None]:
        """Handles streaming tasks using SSE"""
        pass
    

def main():
    # Get agent cards using each agent's static get_agent_card method
    agent_cards = []
    for agent_config in AGENT_CONFIG:
        try:
            # Use the static method to get the card
            agent_card = agent_config["class"].get_agent_card(SERVER_HOST, agent_config["port"])
            agent_cards.append(agent_card)
            logger.info(f"Successfully created {agent_config['type']} agent card for port {agent_config['port']}")
        except Exception as e:
            logger.error(f"Failed to create {agent_config['type']} agent card: {e}")
            continue    
    if not agent_cards:
        logger.error("No agent cards could be created. Check for errors.")
        return
    
    logger.info(f"Starting Multi-Agent Server with {len(agent_cards)} agent cards")
    for i, card in enumerate(agent_cards, 1):
        logger.info(f"{i}. {card.name} - {len(card.skills) if card.skills else 0} skills")

    # Use the first card as the default one for backward compatibility
    server = A2ACardServer(
        host=SERVER_HOST,
        port=SERVER_PORT,
        agent_card=agent_cards[0],
        agent_cards=agent_cards,
        task_manager=DummyTaskManager()
    )
    
    logger.info(f"Multi-Agent Server started at http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Agent Cards available at: http://{SERVER_HOST}:{SERVER_PORT}/.well-known/agents.json")
    
    # Start the server (this call is blocking)
    server.start()

if __name__ == "__main__":
    main()
