#!/usr/bin/env python
"""
Multi-Agent Client - Uses A2ACardsResolver to discover agent cards
"""
import os
import sys
import asyncio
import logging
from typing import List, Dict, Any

# Adjust path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from a2a_agents.a2a_cards_resolver import A2ACardsResolver
from a2a_agents.base_a2a_client import BaseA2AClient
from common.types import AgentCard, Message, TaskSendParams, Task

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL for the multi-agent server
MULTI_AGENT_SERVER_URL = "http://localhost:8888"

async def discover_agents(server_url: str) -> List[AgentCard]:
    """Discover all agent cards from a multi-agent server"""
    logger.info(f"Connecting to multi-agent server at {server_url}")
    resolver = A2ACardsResolver(server_url)
    
    # Get all agent cards
    try:
        agent_cards = resolver.get_agent_cards()
        logger.info(f"Found {len(agent_cards)} agent cards")
        return agent_cards
    except Exception as e:
        logger.error(f"Error discovering agent cards: {e}")
        return []

async def send_test_message(agent_card: AgentCard, message: str) -> Task:
    """Send a test message to an agent"""
    try:
        # Create a client with the agent card
        client = BaseA2AClient(agent_card=agent_card)
        logger.info(f"Sending test message to {agent_card.name} at {agent_card.url}")
        
        # Create a message
        from a2a_agents.a2a_utils.a2a_helper_functions import create_client_message, create_task_params
        client_message = create_client_message(text_content=message)
        task_params = create_task_params(client_message, task_id=f"test_{agent_card.name.lower().replace(' ', '_')}")
        
        # Since the agent isn't actually running, this would fail in a real scenario
        # We're just demonstrating how the client would be set up
        logger.info(f"Would send message: '{message}' to {agent_card.name}")
        logger.info(f"(This is a simulation - no actual request is made since the agent isn't running)")
        
        return None
    except Exception as e:
        logger.error(f"Error sending test message: {e}")
        return None

def print_agent_info(agent_cards: List[AgentCard]):
    """Print information about discovered agents"""
    print("\n" + "="*50)
    print("DISCOVERED AGENTS")
    print("="*50)
    
    for i, card in enumerate(agent_cards, 1):
        print(f"\n{i}. {card.name}")
        print(f"   Description: {card.description}")
        print(f"   URL: {card.url}")
        print(f"   Version: {card.version}")
        
        # Print skills
        if card.skills:
            print(f"   Skills ({len(card.skills)}):")
            for j, skill in enumerate(card.skills, 1):
                print(f"     {j}. {skill.name} (ID: {skill.id})")
                print(f"        Description: {skill.description}")
                if skill.examples:
                    print(f"        Examples: {', '.join(skill.examples)}")
        else:
            print("   Skills: None")
        
        print(f"   Streaming supported: {card.capabilities.streaming if card.capabilities else False}")
    
    print("\n" + "="*50)

async def main():
    print("\nMulti-Agent Client - Agent Discovery Tool")
    print("----------------------------------------\n")
    
    # Get agent cards from the multi-agent server
    agent_cards = await discover_agents(MULTI_AGENT_SERVER_URL)
    
    if not agent_cards:
        print("No agents discovered. Make sure the multi-agent server is running.")
        return
    
    # Print information about discovered agents
    print_agent_info(agent_cards)
    
    # Demonstrate finding a specific agent by name and skill
    try:
        # Find by name
        faq_card = next((card for card in agent_cards if "FAQ" in card.name), None)
        if faq_card:
            print(f"\nFound FAQ agent: {faq_card.name}")
            
            # Demonstrate how to find using the resolver
            resolver = A2ACardsResolver(MULTI_AGENT_SERVER_URL)
            card_by_name = resolver.get_agent_card_by_name(faq_card.name)
            print(f"Verified agent retrieval by name: {card_by_name.name}")
            
            # Demonstrate sending a test message
            await send_test_message(faq_card, "What is your name?")
        
        # Find by skill
        tech_card = next((card for card in agent_cards 
                          if card.skills and any("troubleshooting" in skill.name.lower() for skill in card.skills)), 
                         None)
        if tech_card:
            print(f"\nFound Technical agent by skill: {tech_card.name}")
            skill_id = tech_card.skills[0].id if tech_card.skills else "unknown"
            print(f"Skill ID: {skill_id}")
            
            # Find by skill ID using resolver
            if skill_id != "unknown":
                try:
                    resolver = A2ACardsResolver(MULTI_AGENT_SERVER_URL)
                    card_by_skill = resolver.get_agent_card_by_skill_id(skill_id)
                    print(f"Verified agent retrieval by skill ID: {card_by_skill.name}")
                except ValueError as e:
                    print(f"Could not find card by skill ID: {e}")
    except Exception as e:
        logger.error(f"Error in agent discovery demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
