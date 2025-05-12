import asyncio
import sys
import os
import json

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the client resolver
from a2a_cards_resolver import A2ACardsResolver
from common.client.client import A2AClient

# Example usage of A2ACardsResolver
async def main():
    print("-" * 60)
    print("A2ACardsResolver Example Client")
    print("-" * 60)
    
    # Server URL
    server_url = "http://localhost:8001"
    
    try:
        # Create an instance of A2ACardsResolver
        print(f"Connecting to server at {server_url}...")
        resolver = A2ACardsResolver(server_url)
        
        # Get all agent cards
        print("\nRetrieving all agent cards...")
        agent_cards = resolver.get_agent_cards()
        print(f"Found {len(agent_cards)} agent cards:")
        
        # Display each agent card
        for i, card in enumerate(agent_cards, 1):
            print(f"\n{i}. Agent Card: {card.name}")
            print(f"   Version: {card.version}")
            print(f"   Description: {card.description}")
            print(f"   URL: {card.url}")
            try:
                print(f"   Provider: {card.provider.name}")
            except:
                pass
            print(f"   Number of skills: {len(card.skills) if card.skills else 0}")
            
            # Display skills if available
            if card.skills:
                print("\n   Skills:")
                for j, skill in enumerate(card.skills, 1):
                    print(f"     {j}. {skill.name} ({skill.id})")
                    print(f"        Description: {skill.description}")
        
        # Try to find a card by name (if multiple cards exist)
        if len(agent_cards) > 1:
            try:
                # Get the name of the first card
                first_card_name = agent_cards[0].name
                print(f"\nFinding card with name '{first_card_name}'...")
                card = resolver.get_agent_card_by_name(first_card_name)
                print(f"Found card: {card.name}")
            except ValueError as e:
                print(f"Error finding card by name: {e}")
        
        # Try to find a card by skill ID
        if agent_cards and agent_cards[0].skills:
            try:
                # Get the ID of the first skill of the first card
                first_skill_id = agent_cards[0].skills[0].id
                print(f"\nFinding card with skill ID '{first_skill_id}'...")
                card = resolver.get_agent_card_by_skill_id(first_skill_id)
                print(f"Found card: {card.name}")
            except ValueError as e:
                print(f"Error finding card by skill ID: {e}")

        # Show compatibility with original A2ACardResolver
        print("\nVerifying compatibility with original A2ACardResolver...")
        first_card = resolver.get_agent_card()
        print(f"get_agent_card() returns: {first_card.name}")
                
        # Example of creating an A2AClient with a card
        print("\nCreating A2AClient with the first agent card...")
        client = A2AClient(agent_card=agent_cards[0])
        print(f"Created A2AClient with URL: {client.url}")
        
    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
