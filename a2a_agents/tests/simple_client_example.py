import asyncio
import os
from agents_examples.base_a2a_client import BaseA2AClient
from a2a_utils.a2a_helper_functions import create_task_params

# Configuration
AGENT_URL = "http://localhost:8001"  # Update with your agent's URL

async def main():
    print("=== Simple A2A Client Example ===")
    
    try:
        # Get the agent cards from the URL
        cards = await BaseA2AClient.get_agent_cards_from_url(AGENT_URL)
        if not cards:
            print("No agent cards found!")
            return
        
        # Print available agents
        print(f"Found {len(cards)} agents:")
        for i, card in enumerate(cards):
            print(f"{i+1}. {card.name} - {card.description}")
        
        # Create a client with the first agent card
        client = BaseA2AClient(agent_card=cards[0], output_dir="output")
        
        # Create a message and send it
        message = BaseA2AClient.create_client_message(text_content="Hello! Please tell me what you can do.")
        task_params = create_task_params(message)
        
        # For standard communication:
        print("\nSending message using standard communication...")
        task_id, response = await client.send_task_standard(task_params)
        
        # Display response
        agent_messages = [msg for msg in client.get_task_messages(task_id) if msg.role == "agent"]
        if agent_messages:
            print("\nAgent response:")
            for part in agent_messages[-1].parts:
                if hasattr(part, 'text'):
                    print(part.text)
        
        # For streaming communication:
        print("\nSending message using streaming communication...")
        streaming_message = BaseA2AClient.create_client_message(
            text_content="Please explain your capabilities step by step."
        )
        streaming_task_params = create_task_params(streaming_message)
        task_id, response_stream = await client.send_task_streaming(streaming_task_params)
        
        print("\nStreaming response:")
        async for response in response_stream:
            # Handle different response types
            if hasattr(response.result, 'delta'):
                delta = response.result.delta
                if delta.role == "agent":
                    for part in delta.parts:
                        if hasattr(part, 'text') and part.text:
                            print(part.text, end='', flush=True)
            elif hasattr(response.result, 'status'):
                status = response.result.status
                if status.state == "completed":
                    print("\nTask completed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
