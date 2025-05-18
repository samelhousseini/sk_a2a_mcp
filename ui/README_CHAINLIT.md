# Chainlit UI for A2A Orchestrator

This application provides a web-based user interface for interacting with the Agent-to-Agent (A2A) 
Orchestrator and its specialized agents using Chainlit.

## Prerequisites

- Python 3.9+
- All dependencies from the main project
- Chainlit

## Installation

1. Make sure all the main project dependencies are installed:

```bash
pip install -r requirements.txt
```

2. Install Chainlit:

```bash
pip install chainlit
```

3. Set up environment variables:

Create a `.env` file in the `app` directory with the following variables:

```
ORCHESTRATOR_AGENT_URL=http://localhost:8000
FAQ_AGENT_URL=http://localhost:8001
TECH_AGENT_URL=http://localhost:8002
ESCALATION_AGENT_URL=http://localhost:8003
```

## Running the Application

1. Make sure all the agent servers are running:

```bash
# In separate terminal windows
python app/orchestrator_agent.py
python app/faq_agent.py
python app/technical_troubleshooting_agent.py
python app/human_support_escalation_agent.py
```

2. Run the Chainlit application:

```bash
cd app
chainlit run chat.py -w
```

3. Open your browser and navigate to http://localhost:8000

## Features

- Chat interface for interacting with the A2A Orchestrator
- Support for FAQ queries
- Technical troubleshooting assistance
- Human support escalation
- Streaming responses
- Session management

## Architecture

The application consists of:

1. `orchestrator.py` - A wrapper class for the BaseA2AClient that handles communication with the Orchestrator agent
2. `chat.py` - The Chainlit application that provides the web UI

## How to Use

1. Start by asking a question like "What are your support hours?"
2. For technical issues, describe your problem: "My internet is very slow"
3. To escalate to human support: "I need to speak to a human agent"
