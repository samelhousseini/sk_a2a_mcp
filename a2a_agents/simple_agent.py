from a2a_agents.sk_a2a_agent import SK_A2A_Agent
from common.types import AgentSkill, AgentCapabilities, AgentCard
import logging

from typing import List, Dict, Any, Optional, AsyncIterable, Union, Callable, Tuple, ClassVar

from rich.console import Console
console = Console()



class SimpleAgent(SK_A2A_Agent):

    # This method is to be overridden in subclasses
    async def formulate_response(self, task_id: str, agent_query: Dict[str, Any]) -> Any:
        '''Formulates a response based on the agent query.'''
        
        thread = await self.get_chat_history_thread(task_id)

        console.print(f"Sending agent query for task {task_id}: {agent_query}")
        response = await self.agent.get_response(message=agent_query, thread=thread)
        console.print(f"Received response for task {task_id}: {response.content}")

        return response


