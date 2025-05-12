import logging

import click

from dotenv import load_dotenv
from common.types import AgentSkill, AgentCapabilities, AgentCard
from common.server import A2AServer
from a2a_agents.simple_agent import SimpleAgent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=8000) 
def main(host, port):
    skill = AgentSkill(
        id="my-project-echo-skill",
        name="Echo Tool",
        description="Echos the input given",
        tags=["echo", "repeater"],
        examples=["I will see this echoed back to me"],
        inputModes=["text/plain", "application/json"],
        outputModes=["text/plain", "application/json"],
    )
    logging.info(skill)

    capabilities = AgentCapabilities(
        streaming=True
    )
    agent_card = AgentCard(
        name="Echo Agent",
        description="This agent echos the input given",
        url=f"http://{host}:{port}/",
        version="0.1.0",
        defaultInputModes=["text/plain", "application/json"],
        defaultOutputModes=["text/plain", "application/json"],
        capabilities=capabilities,
        skills=[skill]
    )
    logging.info(agent_card)

    task_manager = SimpleAgent()
    server = A2AServer(
        agent_card=agent_card,
        task_manager=task_manager,
        host=host,
        port=port,
    )
    server.start()




if __name__ == "__main__":
    main()
