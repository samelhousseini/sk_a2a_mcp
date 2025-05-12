import json
import logging


from typing import Any, List
from starlette.requests import Request
from starlette.responses import JSONResponse


from common.server.task_manager import TaskManager
from common.types import (
    A2ARequest,
    AgentCard,
)

from common.server import A2AServer

logger = logging.getLogger(__name__)



class A2ACardServer(A2AServer):
    """
    Extended A2A Server that can handle multiple agent cards.
    Adds support for the /.well-known/agents.json endpoint which returns a list of all agent cards.
    """
    
    def __init__(
        self,
        host='0.0.0.0',
        port=5000,
        endpoint='/',
        agent_card: AgentCard = None,
        agent_cards: List[AgentCard] = None,
        task_manager: TaskManager = None,
    ):
        # Initialize with the base A2AServer
        super().__init__(
            host=host,
            port=port,
            endpoint=endpoint,
            agent_card=agent_card,
            task_manager=task_manager,
        )
        
        # Store the list of agent cards
        self.agent_cards = agent_cards or []
        
        # If agent_card is provided but agent_cards is not, add it to the list
        if agent_card and not agent_cards:
            self.agent_cards = [agent_card]
        
        # Add the new route for multiple agent cards
        self.app.add_route(
            '/.well-known/agents.json', self._get_agent_cards, methods=['GET']
        )

    def _get_agent_cards(self, request: Request) -> JSONResponse:
        """Return a list of all agent cards."""
        # Convert each agent card to dict and exclude None values
        agent_cards_data = [card.model_dump(exclude_none=True) for card in self.agent_cards]
        return JSONResponse(agent_cards_data)
    
    def start(self):
        """Override start to check for agent_cards instead of just agent_card."""
        if not self.agent_cards:
            raise ValueError('No agent cards defined')

        if self.task_manager is None:
            raise ValueError('request_handler is not defined')

        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
