"""
AI Knowledge Agent — Main Entry Point.
Uses a LangChain ReAct agent with tools to answer questions
from documents stored in Pinecone.
"""

import argparse
import sys

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from config import Config
from tools import vector_search_tool, summarize_tool, multi_query_search_tool

console = Console()

# ReAct Agent Prompt
AGENT_PROMPT = PromptTemplate.from_template(
    """You are an intelligent knowledge assistant. You help users find and understand 
information from their document knowledge base stored in a Pinecone vector database.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Guidelines:
- Always start with `vector_search_tool` to find relevant information.
- Use `summarize_tool` if you get long or multiple search results that need condensing.
- Use `multi_query_search_tool` if the initial search doesn't return good results,
  or if the question is complex and could be interpreted multiple ways.
- Always cite your sources in the final answer (include document name and chunk number).
- If no relevant information is found, say so honestly.
- Keep your final answer well-structured and informative.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


def create_agent() -> AgentExecutor:
    """Create and configure the ReAct agent with tools."""
    Config.validate()

    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
        temperature=0,
    )

    tools = [vector_search_tool, summarize_tool, multi_query_search_tool]

    agent = create_react_agent(llm=llm, tools=tools, prompt=AGENT_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )


def format_agent_response(result: dict) -> str:
    """Format the agent's response with tool usage info."""
    output_parts = []

    # Show intermediate steps (tools used)
    if result.get("intermediate_steps"):
        output_parts.append("[dim]🧠 Agent reasoning:[/dim]")
        for action, observation in result["intermediate_steps"]:
            tool_name = action.tool
            output_parts.append(f"   [cyan]→ Used tool:[/cyan] [bold]{tool_name}[/bold]")
        output_parts.append("")

    # Final answer
    output_parts.append(result.get("output", "No answer generated."))

    return "\n".join(output_parts)


def interactive_mode(agent_executor: AgentExecutor) -> None:
    """Run the agent in interactive chat mode."""
    console.print(
        Panel(
            "[bold cyan]🤖 AI Knowledge Agent — Interactive Mode[/bold cyan]\n"
            "Ask questions about your ingested documents.\n"
            "Type [bold]'quit'[/bold] to exit, [bold]'help'[/bold] for tips.",
            border_style="cyan",
        )
    )
    console.print()

    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                console.print("[dim]👋 Goodbye![/dim]")
                break

            if query.lower() == "help":
                console.print(
                    Panel(
                        "💡 [bold]Tips:[/bold]\n"
                        "• Ask specific questions about your documents\n"
                        '• Try: "What are the main topics covered?"\n'
                        '• Try: "Summarize the key findings"\n'
                        '• Try: "Compare concept A with concept B"\n'
                        "• Type 'quit' to exit",
                        border_style="yellow",
                    )
                )
                continue

            # Run the agent
            console.print("[dim]🧠 Thinking...[/dim]\n")

            result = agent_executor.invoke({"input": query})
            formatted = format_agent_response(result)

            console.print(
                Panel(
                    formatted,
                    title="[bold]🤖 Answer[/bold]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]👋 Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]\n")


def single_query_mode(agent_executor: AgentExecutor, query: str) -> None:
    """Run a single query and print the result."""
    console.print(f"[bold green]Query:[/bold green] {query}\n")
    console.print("[dim]🧠 Thinking...[/dim]\n")

    try:
        result = agent_executor.invoke({"input": query})
        formatted = format_agent_response(result)
        console.print(
            Panel(
                formatted,
                title="[bold]🤖 Answer[/bold]",
                border_style="green",
                padding=(1, 2),
            )
        )
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Knowledge Agent — Ask questions about your documents"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query instead of interactive mode",
    )
    args = parser.parse_args()

    agent_executor = create_agent()

    if args.query:
        single_query_mode(agent_executor, args.query)
    else:
        interactive_mode(agent_executor)


if __name__ == "__main__":
    main()
