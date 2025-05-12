import json
from typing import List

import httpx

from common.types import (
    A2AClientJSONError,
    AgentCard,
)


from common.client import A2ACardResolver


class A2ACardsResolver(A2ACardResolver):
    """
    Client resolver for handling multiple agent cards from an A2ACardServer.
    Connects to the /.well-known/agents.json endpoint to retrieve all available agent cards.
    """
    
    def __init__(self, base_url, agent_cards_path='/.well-known/agents.json'):
        """
        Initialize the resolver with the base URL and path to the agent cards endpoint.
        
        Args:
            base_url: The base URL of the A2ACardServer
            agent_cards_path: Path to the agents.json endpoint (default: '/.well-known/agents.json')
        """
        # Call the parent constructor with the base URL and original path
        # This ensures get_agent_card() still works for backward compatibility
        super().__init__(base_url)
        
        # Store the agent cards path
        self.base_url = base_url.rstrip('/')
        self.agent_cards_path = agent_cards_path.lstrip('/')
    
    def get_agent_cards(self) -> List[AgentCard]:
        """
        Retrieve all agent cards from the server.
        
        Returns:
            List[AgentCard]: A list of agent card objects
            
        Raises:
            A2AClientJSONError: If the server response cannot be parsed as JSON
            httpx.HTTPError: If the HTTP request fails
        """
        with httpx.Client() as client:
            # Get the list of agent cards from the server
            response = client.get(self.base_url + '/' + self.agent_cards_path)
            response.raise_for_status()
            
            try:
                # Parse the JSON response
                cards_data = response.json()
                
                # Verify we received a list
                if not isinstance(cards_data, list):
                    raise A2AClientJSONError(f"Expected a list of agent cards, got {type(cards_data)}")
                
                # Convert each item to an AgentCard object
                agent_cards = [AgentCard(**card_data) for card_data in cards_data]
                return agent_cards
                
            except json.JSONDecodeError as e:
                raise A2AClientJSONError(f"Failed to parse JSON response: {str(e)}") from e
    
    def get_agent_card_by_name(self, name: str) -> AgentCard:
        """
        Find an agent card by name from the list of available agent cards.
        
        Args:
            name: The name of the agent card to find
            
        Returns:
            AgentCard: The agent card with the specified name
            
        Raises:
            ValueError: If no agent card with the specified name is found
        """
        agent_cards = self.get_agent_cards()
        
        for card in agent_cards:
            if card.name == name:
                return card
        
        raise ValueError(f"No agent card found with name: {name}")
    
    def get_agent_card_by_skill_id(self, skill_id: str) -> AgentCard:
        """
        Find an agent card that has a skill with the specified ID.
        
        Args:
            skill_id: The ID of the skill to search for
            
        Returns:
            AgentCard: The first agent card that has a skill with the specified ID
            
        Raises:
            ValueError: If no agent card with the specified skill ID is found
        """
        agent_cards = self.get_agent_cards()
        
        for card in agent_cards:
            if card.skills:
                for skill in card.skills:
                    if skill.id == skill_id:
                        return card
        
        raise ValueError(f"No agent card found with skill ID: {skill_id}")
