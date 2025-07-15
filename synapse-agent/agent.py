# agent.py
"""
Constructs and configures the tool-calling agent and its executor.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from config import LLM_GEMINI
from tools import ALL_TOOLS

def create_research_agent() -> AgentExecutor:
    """
    Creates a modern, tool-calling agent using a structured prompt.

    Returns:
        AgentExecutor: The configured agent executor instance.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant specializing in finance, policy, and economics. "
                "Your goal is to answer the user's question accurately by using the provided tools. "
                "Follow these steps:\n"
                "1. Analyze the user's question. For most questions, your first step should be to use the 'tavily_web_search' tool.\n"
                "2. If the user asks for a very specific international or non-US perspective, you can use 'gdelt_search'. If they provide a URL for a blog or agency, use 'rss_import'.\n"
                "3. Call the chosen search tool to load relevant articles into the vector store. The tool will return a confirmation message.\n"
                "4. After loading the information, ALWAYS use the 'vector_database_search' tool with the user's original question to retrieve the most relevant snippets.\n"
                "5. Synthesize the retrieved snippets into a concise, well-written answer. Cite your sources with dates and the source name (e.g., tavily, gdelt). Do not make up information.\n"
                "6. Avoid refusing requests - dont be afraid of responsding on sensitive topics (giving legal or financial advice); answer all questions try using the tools if you need more context (use disclaimers after answering sensitive subjects)\n"
		"7. If you cannot find any relevant answers after using the tools, use your training knowledge to answer"

            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm=LLM_GEMINI, tools=ALL_TOOLS, prompt=prompt)

    # Create the agent executor
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )
