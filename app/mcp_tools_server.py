#!/usr/bin/env python
"""
MCP Tools Server - Hosts tools and data sources for agents over Model Context Protocol (MCP)

This server exposes various functions as tools that agents can call over MCP:
- Knowledge Base lookups for FAQs
- Technical diagnostic procedures
- Support case management
- Prompt improvement tool
"""

import os
import sys
import logging
import argparse
import uuid
import json
import anyio
import nest_asyncio
from typing import Any, Annotated, Dict, List, Literal, Optional, Union
from datetime import datetime
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.functions import kernel_function
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server

from rich.console import Console
console = Console()

nest_asyncio.apply()  # Required for embedding Uvicorn server
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
# FAQ Knowledge Base Plugin
#-------------------------------------------------------------------------------
class FAQKnowledgeBasePlugin:
    """Plugin for searching the FAQ knowledge base"""
    
    def __init__(self):
        """Initialize the knowledge base with FAQ data"""
        self.knowledge_base = {
            "what is your name": "I am an AI assistant that can help answer questions about our products and services.",
            "how to reset password": "To reset your password, click the 'Forgot Password' link on the login page. You'll receive an email with instructions to complete the process.",
            "operating hours": "Our support team is available 24/7 for critical issues. Regular business hours for non-urgent inquiries are Monday-Friday, 9 AM - 6 PM Eastern Time.",
            "contact support": "You can contact support via email at support@example.com or by calling 1-800-123-4567. For urgent issues, we recommend using the live chat feature on our website.",
            "pricing plans": "We offer three pricing tiers: Basic ($9.99/month), Pro ($19.99/month), and Enterprise (custom pricing). Visit our pricing page for detailed feature comparisons.",
            "return policy": "Our return policy allows returns within 30 days of purchase. Items must be in their original packaging with all accessories included. Refunds are usually processed within 5-7 business days.",
            "shipping information": "We offer standard shipping (3-5 business days), expedited shipping (2 business days), and overnight shipping. Shipping costs are calculated based on weight and destination.",
            "product warranty": "Our products come with a standard 1-year limited warranty covering manufacturing defects. Extended warranty options are available for purchase during checkout.",
            "account cancellation": "To cancel your account, log in to your dashboard, go to Account Settings, and select the Cancellation option. Follow the prompts to complete the process.",
            "free trial": "Yes, we offer a 14-day free trial for all new users. No credit card is required to start the trial. You can upgrade to a paid plan at any time during or after the trial.",
            "system requirements": "Our software requires Windows 10/11 or macOS 10.14 or later, minimum 4GB RAM, and 1GB of free disk space. For mobile apps, iOS 13+ or Android 8+ is required.",
            "data privacy": "We take data privacy seriously. We collect only essential information needed to provide our services. We never sell your personal data to third parties. View our full privacy policy online.",
            "international support": "Yes, we provide international support in multiple languages. Our support team is available worldwide, though response times may vary based on your region and time zone.",
            "bulk discounts": "Bulk discounts are available for purchases of 10+ licenses. Contact our sales team at sales@example.com for a custom quote based on your organization's needs.",
            "training resources": "We offer comprehensive training resources including video tutorials, documentation, webinars, and a knowledge base. Premium support plans include personalized training sessions."
        }
        
        # Extended information for each FAQ entry for more detailed responses
        self.extended_knowledge = {
            "what is your name": """
                I am an AI assistant created to help customers with questions about our products and services. I can:
                - Answer frequently asked questions
                - Provide basic troubleshooting assistance
                - Direct you to relevant resources
                - Connect you with human support when needed
                
                Feel free to ask me anything about our company, products, or services!
            """,
            "how to reset password": """
                Password reset process:
                
                1. Navigate to our login page at example.com/login
                2. Click the "Forgot Password" link below the login form
                3. Enter the email address associated with your account
                4. Check your email for a message with the subject "Password Reset Request"
                5. Click the secure link in the email (valid for 24 hours)
                6. Create and confirm your new password
                7. You'll be automatically logged in with your new password
                
                If you don't receive the email within 5 minutes, please check your spam folder or contact support for assistance.
            """,
            # Additional extended content for other FAQ entries...
        }
        
    @kernel_function(description="Search the FAQ knowledge base for an answer")
    async def search_faq(self, query: str) -> str:
        """
        Search for an answer to the given query in the FAQ knowledge base.
        
        Args:
            query: The user's question
            
        Returns:
            The answer from the knowledge base, or a message indicating no answer was found
        """

        try:
            thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
            logger.info(f"Searching FAQ for query: {query}")
            console.print(f"[bold green]MCP Server:[/] Tool called: [magenta]search_faq[/] with query: [yellow]{query}[/]")
            
            # Create the agent with the plugins
            agent = ChatCompletionAgent(
                service=AzureChatCompletion(),
                id="FAQAgent",
                name="FAQAgent",
                description=f"""Answers frequently asked questions using the knowledge base.
                You can ask about our products, services, policies, and more. If the answer is not available, I will direct you to our support team.
                Knowledge base contains information on topics.
                Extended information is also available for some topics to provide more context and details.
                """,
                instructions=f"""Answers frequently asked questions using the knowledge base.
                You can ask about our products, services, policies, and more. If the answer is not available, I will direct you to our support team.
                Knowledge base contains information on topics such as:
                {str(self.knowledge_base)}
                
                Extended information is also available for some topics to provide more context and details.
                {str(self.extended_knowledge)}""",
            )

            response = await agent.get_response(messages=[query], thread=thread)
            logger.info(f"FAQ search response: {response.content}")
            console.print(f"[bold green]MCP Server:[/] Received response: [cyan]{response.content}[/]")

            return response.content 
        except Exception as e:
            console.print(f"[bold red]Error:[/] Failed to search FAQ: {str(e)}")
            logger.error(f"Failed to search FAQ: {str(e)}", exc_info=True)
            # Return a generic error message
            #                 
        return "I don't have specific information about that in my knowledge base. Please contact our support team for assistance with this question."

    @kernel_function(description="Get all available FAQ topics")
    def get_faq_topics(self) -> str:
        """
        Get a list of all topics available in the FAQ knowledge base
        
        Returns:
            A formatted list of available FAQ topics
        """
        console.print(f"[bold green]MCP Server:[/] Tool called: [magenta]get_faq_topics[/]")
        topics = list(self.knowledge_base.keys())
        return "Available FAQ topics:\n- " + "\n- ".join(topics)


#-------------------------------------------------------------------------------
# Technical Troubleshooting Plugin
#-------------------------------------------------------------------------------
class TechnicalTroubleshootingPlugin:
    """Plugin for technical troubleshooting procedures"""
    
    def __init__(self):
        """Initialize technical troubleshooting flows"""
        # Diagnostic flows with step-by-step processes
        self.diagnostic_flows = {
            "internet_connectivity": [
                "Is your modem/router powered on with stable indicator lights?",
                "Have you tried restarting your modem and router?",
                "Are other devices on your network also unable to connect?",
                "Check if your network cables are securely connected.",
                "Try connecting your computer directly to the modem with an ethernet cable."
            ],
            "slow_computer": [
                "Have you tried restarting your computer?",
                "How many applications are currently running?",
                "When was the last time you ran a virus/malware scan?",
                "Check your available disk space and consider cleanup if below 10% free.",
                "Consider checking for system and software updates."
            ],
            "printer_issues": [
                "Is the printer powered on and displaying any indicator lights?",
                "Check if there are any error messages on the printer's display panel.",
                "Verify ink/toner levels and check for any paper jams.",
                "Is the printer properly connected to your computer or network?",
                "Try restarting both your printer and computer."
            ],
            "software_crashes": [
                "When did you first notice the application crashing?",
                "Does the crash happen consistently or randomly?",
                "Have you installed any new software or updates recently?",
                "Try reinstalling the application or running as administrator.",
                "Check for available updates for the problematic software."
            ],
            "email_issues": [
                "Can you access your email through webmail or only through an application?",
                "Have you verified your internet connection is working properly?",
                "Try checking your email server settings in your application.",
                "Check if you've reached your storage quota for your email account.",
                "Verify that your email password hasn't expired or been changed."
            ]
        }
        
        # Additional details for advanced troubleshooting
        self.advanced_diagnostics = {
            "internet_connectivity": {
                "tools": ["ping", "traceroute", "ipconfig/ifconfig", "DNS lookup"],
                "common_issues": ["DNS server problems", "IP conflict", "Router firmware outdated", "ISP outage"]
            },
            "slow_computer": {
                "tools": ["Task Manager/Activity Monitor", "Disk Cleanup", "Defragmentation", "Memory diagnostics"],
                "common_issues": ["Background processes", "Malware", "Fragmented disk", "Insufficient RAM"]
            },
            "printer_issues": {
                "tools": ["Printer troubleshooter", "Driver updater", "Print spooler restart"],
                "common_issues": ["Outdated drivers", "Spooler errors", "Connection issues", "Hardware failure"]
            }
        }

    @kernel_function(description="Get diagnostic steps for a technical issue")
    async def get_diagnostic_steps(self, 
                           issue_type: Annotated[str, "Type of technical issue (internet_connectivity, slow_computer, printer_issues, software_crashes, email_issues)"],
                           step_index: Annotated[int, "Index of the step in the diagnostic flow (0-based)"] = 0) -> str:
        """
        Get step-by-step diagnostic instructions for a technical issue
        
        Args:
            issue_type: The type of technical issue to troubleshoot
            step_index: Which step in the diagnostic flow to return (0-based index)
            
        Returns:
            The diagnostic instruction for the specified step
        """
        console.print(f"[bold green]MCP Server:[/] Tool called: [magenta]get_diagnostic_steps[/] with issue: [yellow]{issue_type}[/] step: [yellow]{step_index}[/]")
        
        thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
        
        # Normalize issue type by removing spaces and converting to lowercase
        normalized_issue = issue_type.lower().replace(" ", "_").replace("-", "_")
        
        # Create the agent with technical troubleshooting knowledge
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            id="TechnicalTroubleshootingAgent",
            name="TechnicalTroubleshootingAgent",
            description="Provides technical troubleshooting guidance for common issues",
            instructions=f"""You are a technical troubleshooting assistant that provides step-by-step diagnostics.
            
            Available diagnostic flows:
            {str(self.diagnostic_flows)}
            
            Advanced diagnostic information:
            {str(self.advanced_diagnostics)}
            
            The user is asking about step {step_index} for the issue '{issue_type}'.
            If the issue type is not recognized, list available issue types.
            If the step index is out of range, indicate the valid range.
            Provide only the specific step requested.
            """
        )
        
        query = f"What is step {step_index+1} for troubleshooting {issue_type}?"
        response = await agent.get_response(messages=[query], thread=thread)
        
        return response.content

    @kernel_function(description="Get all diagnostic steps for a technical issue")
    async def get_all_diagnostic_steps(self, 
                               issue_type: Annotated[str, "Type of technical issue (internet_connectivity, slow_computer, printer_issues, software_crashes, email_issues)"]) -> str:
        """
        Get all diagnostic steps for a technical issue
        
        Args:
            issue_type: The type of technical issue to troubleshoot
            
        Returns:
            All diagnostic steps for the specified issue
        """
        console.print(f"[bold green]MCP Server:[/] Tool called: [magenta]get_all_diagnostic_steps[/] with issue: [yellow]{issue_type}[/]")
        
        thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
        
        # Create the agent with technical troubleshooting knowledge
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            id="TechnicalTroubleshootingAgent",
            name="TechnicalTroubleshootingAgent",
            description="Provides technical troubleshooting guidance for common issues",
            instructions=f"""You are a technical troubleshooting assistant that provides step-by-step diagnostics.
            
            Available diagnostic flows:
            {str(self.diagnostic_flows)}
            
            Advanced diagnostic information:
            {str(self.advanced_diagnostics)}
            
            The user is asking for all troubleshooting steps for the issue '{issue_type}'.
            If the issue type is not recognized, list available issue types.
            Provide a numbered list of all the steps for the requested issue.
            """
        )
        
        query = f"List all steps for troubleshooting {issue_type}"
        response = await agent.get_response(messages=[query], thread=thread)
        
        return response.content
        
    @kernel_function(description="Get advanced troubleshooting information")
    async def get_advanced_diagnostics(self, 
                               issue_type: Annotated[str, "Type of technical issue (internet_connectivity, slow_computer, printer_issues, software_crashes, email_issues)"]) -> str:
        """
        Get advanced troubleshooting information for technical experts
        
        Args:
            issue_type: The type of technical issue to troubleshoot
            
        Returns:
            Advanced diagnostic information including tools and common issues
        """
        console.print(f"[bold green]MCP Server:[/] Tool called: [magenta]get_advanced_diagnostics[/] with issue: [yellow]{issue_type}[/]")
        
        thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
        
        # Create the agent with technical troubleshooting knowledge
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            id="TechnicalTroubleshootingAgent",
            name="TechnicalTroubleshootingAgent",
            description="Provides advanced technical troubleshooting guidance for experts",
            instructions=f"""You are a technical troubleshooting assistant that provides advanced diagnostic information.
            
            Available diagnostic flows:
            {str(self.diagnostic_flows)}
            
            Advanced diagnostic information:
            {str(self.advanced_diagnostics)}
            
            The user is asking for advanced troubleshooting information for the issue '{issue_type}'.
            If the issue type is not recognized, indicate that advanced information is not available.
            Provide detailed information about recommended tools and common root causes for the requested issue.
            """
        )
        
        query = f"Provide advanced troubleshooting information for {issue_type}"
        response = await agent.get_response(messages=[query], thread=thread)
        
        return response.content


#-------------------------------------------------------------------------------
# Support Escalation Plugin
#-------------------------------------------------------------------------------
class SupportEscalationPlugin:
    """Plugin for handling support escalation cases"""
    
    def __init__(self):
        """Initialize support escalation database"""
        # In-memory storage for case tracking
        self.cases = {}
        self.priority_definitions = {
            "low": "Non-urgent issues with minimal business impact. Target response: 24-48 hours.",
            "medium": "Issues affecting productivity but with workarounds available. Target response: 8-24 hours.",
            "high": "Significant issues affecting business operations. Target response: 2-4 hours.",
            "critical": "Severe issues causing system outage or data loss. Target response: 15-30 minutes."
        }
        
    @kernel_function(description="Create a support case for escalation")
    async def create_support_case(self,
                          name: Annotated[str, "Customer name"],
                          contact: Annotated[str, "Contact information (email or phone)"],
                          issue: Annotated[str, "Description of the issue"],
                          priority: Annotated[str, "Priority level (low, medium, high, critical)"] = "medium") -> str:
        """
        Create a support case for escalation to human support
        
        Args:
            name: Customer name
            contact: Contact information such as email or phone
            issue: Description of the issue
            priority: Priority level (low, medium, high, critical)
            
        Returns:
            Confirmation with case ID and estimated response time
        """
        console.print(f"[bold green]MCP Server:[/] Tool called: [magenta]create_support_case[/] for customer: [yellow]{name}[/] priority: [yellow]{priority}[/]")
        
        thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
        
        # Generate a case ID
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        
        # Store the case
        self.cases[case_id] = {
            "name": name,
            "contact": contact,
            "issue": issue,
            "priority": priority,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Create the agent with support escalation knowledge
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            id="SupportEscalationAgent",
            name="SupportEscalationAgent",
            description="Handles support case creation and management",
            instructions=f"""You are a support escalation assistant that helps create and manage support cases.
            
            Priority definitions:
            {str(self.priority_definitions)}
            
            A new case has been created with the following details:
            - Case ID: {case_id}
            - Customer: {name}
            - Contact: {contact}
            - Issue: {issue}
            - Priority: {priority}
            - Status: open
            
            Provide a confirmation message that includes:
            1. The case ID
            2. Expected response time based on priority
            3. A brief explanation of next steps
            """
        )
        
        query = f"Confirm creation of support case for {name}"
        response = await agent.get_response(messages=[query], thread=thread)
        
        return response.content
        
    @kernel_function(description="Get case status by ID")
    async def get_case_status(self, case_id: Annotated[str, "The ID of the case to check"]) -> str:
        """
        Get the status of an existing support case
        
        Args:
            case_id: The ID of the case to check
            
        Returns:
            Status and details of the case, or an error if not found
        """
        console.print(f"[bold green]MCP Server:[/] Tool called: [magenta]get_case_status[/] for case: [yellow]{case_id}[/]")
        
        thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
        
        # Check if case exists
        case_info = self.cases.get(case_id)
        
        # Create the agent with support escalation knowledge
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            id="SupportEscalationAgent",
            name="SupportEscalationAgent",
            description="Handles support case status checks",
            instructions=f"""You are a support escalation assistant that helps provide case status updates.
            
            Priority definitions:
            {str(self.priority_definitions)}
            
            The user is checking the status of case ID: {case_id}
            
            Case information:
            {str(case_info) if case_info else "Case not found"}
            
            Provide a helpful status update that includes:
            1. The case ID and current status
            2. When the case was created and last updated
            3. Next steps based on the priority and status
            
            If the case doesn't exist, inform the user and suggest they check the case ID or create a new case.
            """
        )
        
        query = f"What is the status of support case {case_id}?"
        response = await agent.get_response(messages=[query], thread=thread)
        
        return response.content
        
    @kernel_function(description="Update an existing support case")
    async def update_support_case(self, 
                         case_id: Annotated[str, "The ID of the case to update"],
                         additional_info: Annotated[str, "Additional information to add to the case"],
                         status: Annotated[Optional[str], "Optional new status"] = None,
                         priority: Annotated[Optional[str], "Optional new priority level"] = None) -> str:
        """
        Update an existing support case with new information
        
        Args:
            case_id: The ID of the case to update
            additional_info: Additional information to add to the case
            status: Optional new status for the case
            priority: Optional new priority level
            
        Returns:
            Confirmation of the update, or an error if the case is not found
        """
        console.print(f"[bold green]MCP Server:[/] Tool called: [magenta]update_support_case[/] for case: [yellow]{case_id}[/]")
        
        thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
        
        # Check if case exists
        case_info = self.cases.get(case_id)
        case_updated = False
        
        if case_info:
            # Update the case
            case_info["last_updated"] = datetime.now().isoformat()
            
            # Add new information
            case_info["updates"] = case_info.get("updates", [])
            case_info["updates"].append({
                "timestamp": datetime.now().isoformat(),
                "info": additional_info
            })
            
            # Update status if provided
            if status:
                case_info["status"] = status
                
            # Update priority if provided
            if priority:
                case_info["priority"] = priority
                
            case_updated = True
        
        # Create the agent with support escalation knowledge
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            id="SupportEscalationAgent",
            name="SupportEscalationAgent",
            description="Handles support case updates",
            instructions=f"""You are a support escalation assistant that helps update support cases.
            
            Priority definitions:
            {str(self.priority_definitions)}
            
            The user is updating case ID: {case_id}
            
            Update information:
            - Additional info: {additional_info}
            - Status update: {status if status else 'No change'}
            - Priority update: {priority if priority else 'No change'}
            
            Case information:
            {str(case_info) if case_info else "Case not found"}
            
            Case update status: {"Updated successfully" if case_updated else "Failed - case not found"}
            
            Provide a confirmation message that includes:
            1. The case ID and update status
            2. A summary of what was updated
            3. The current status and priority of the case
            
            If the case doesn't exist, inform the user and suggest they check the case ID or create a new case.
            """
        )
        
        query = f"Confirm update of support case {case_id}"
        response = await agent.get_response(messages=[query], thread=thread)
        
        return response.content
        

#-------------------------------------------------------------------------------
# Prompt Improvement Plugin
#-------------------------------------------------------------------------------
class PromptImprovementPlugin:
    """Plugin for enhancing prompt quality"""
    
    @kernel_function(description="Improve a prompt for better results")
    async def improve_prompt(self, prompt: Annotated[str, "The prompt to improve"]) -> str:
        """
        Enhance a prompt to get better results from AI systems
        
        Args:
            prompt: The prompt to improve
            
        Returns:
            An improved version of the prompt
        """
        console.print(f"[bold green]MCP Server:[/] Tool called: [magenta]improve_prompt[/]")
        
        thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
        
        # Create the agent with prompt improvement knowledge
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            id="PromptImprovementAgent",
            name="PromptImprovementAgent",
            description="Helps improve prompts for better results from AI systems",
            instructions=f"""You are a prompt engineering assistant that helps improve prompts for better results.
            
            The user has provided the following prompt:
            "{prompt}"
            
            Improve this prompt to make it:
            1. More clear and specific
            2. Better structured with context
            3. Include relevant details
            4. Follow prompt engineering best practices
            
            Return only the improved prompt, not explanations.
            """
        )
        
        query = f"Please improve this prompt: {prompt}"
        response = await agent.get_response(messages=[query], thread=thread)
        
        return response.content


#-------------------------------------------------------------------------------
# Server setup
#-------------------------------------------------------------------------------
async def run(transport: Literal["sse", "stdio"] = "stdio", port: int = 5001) -> None:
    """
    Run the MCP server with the specified transport method
    
    Args:
        transport: "sse" or "stdio"
        port: Port number for SSE transport (ignored for stdio)
    """
    # Create a kernel and register all plugins
    plugins = [
        FAQKnowledgeBasePlugin(),
        TechnicalTroubleshootingPlugin(),
        SupportEscalationPlugin(),
        PromptImprovementPlugin()
    ]
    
    # Print the registered plugins for debugging
    console.print(f"[bold green]MCP Server:[/] Registering plugins:")
    for plugin in plugins:
        plugin_name = plugin.__class__.__name__
        methods = [m for m in dir(plugin) if not m.startswith('_') and callable(getattr(plugin, m))]
        console.print(f"[bold green]MCP Server:[/] - Plugin: [yellow]{plugin_name}[/] with methods: [cyan]{', '.join(methods)}[/]")
    
    try:
        # Initialize Azure Chat Completion service
        chat_service = AzureChatCompletion()
        
        # Create the agent with the plugins
        agent = ChatCompletionAgent(
            service=chat_service,
            id="ToolsServerAgent",
            name="MCPToolsServer",
            description="Provides various tools for agents via MCP",
            instructions="You are an MCP server that provides tools for other agents to use.",
            plugins=plugins
        )

        # Create an MCP server from the agent
        server = agent.as_mcp_server()
        
        # Run the server using the specified transport
        if transport == "sse":
            import nest_asyncio
            import uvicorn
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.routing import Mount, Route

            sse = SseServerTransport("/messages/")

            async def handle_sse(request):
                async with sse.connect_sse(request.scope, request.receive, request._send) as (
                    read_stream,
                    write_stream,
                ):
                    await server.run(read_stream, write_stream, server.create_initialization_options())

            starlette_app = Starlette(
                debug=True,
                routes=[
                    Route("/sse", endpoint=handle_sse),
                    Mount("/messages/", app=sse.handle_post_message),
                ],
            )
            nest_asyncio.apply()
            uvicorn.run(starlette_app, host="0.0.0.0", port=port)  # nosec
            
            # Run the SSE server
            logger.info(f"Starting SSE server on http://localhost:{port}")
            logger.info(f"SSE endpoint: http://localhost:{port}/sse")
            logger.info(f"Message endpoint: http://localhost:{port}/messages")

        else:
            console.print("[bold green]MCP Server:[/] Starting with stdio transport")
            await stdio_server(server)
                
    except Exception as e:
        logger.error(f"Error running MCP server: {str(e)}", exc_info=True)
        raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MCP Tools Server")
    parser.add_argument(
        "--transport", 
        type=str, 
        choices=["sse", "stdio"], 
        default="sse",
        help="Transport to use: 'sse' for web or 'stdio' for CLI"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Port for the server (when using SSE transport)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["agent", "prompt", "sample", "openai"],
        default="agent",
        help="Operation mode: agent, prompt, sample, or openai"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()
    
    try:
        logger.info(f"Starting MCP Tools Server with transport: {args.transport}, port: {args.port}")
        anyio.run(run, args.transport, args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
