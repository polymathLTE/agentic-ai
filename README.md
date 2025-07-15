Of course. Here is a professional and comprehensive `README.md` file for the Synapse project. It is structured for clarity and avoids any informalities or emojis.

---

# Synapse: A Multi-Agent System for Financial Research

Synapse is a collaborative AI system designed to automate complex financial and policy research. It employs a team of specialized AI agents, orchestrated by the LangGraph framework, to handle multi-step queries that are beyond the scope of traditional Retrieval-Augmented Generation (RAG) systems. By decomposing a user's request, gathering information from diverse real-time sources, and synthesizing a comprehensive report, Synapse delivers more reliable, detailed, and factually grounded analysis.

## Table of Contents

- [The Problem](#the-problem)
- [System Architecture](#system-architecture)
  - [The Agent Team](#the-agent-team)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [Example Usage](#example-usage)
- [Future Work](#future-work)
- [License](#license)

## The Problem

Standard Retrieval-Augmented Generation (RAG) systems are effective for direct question-answering but often fail when faced with complex queries that require planning, decomposition, and analysis. A query like, "How has a company's recent earnings call affected its stock price and news coverage?" involves multiple distinct cognitive steps. A single-agent system struggles to manage this process, often resulting in generic or incomplete answers.

Synapse solves this by modeling a human research workflow. It dedicates specialized agents to each phase of the process—planning, data gathering, and analysis—to ensure that every part of a complex query is handled thoroughly and methodically.

## System Architecture

Synapse is built as a stateful graph using the LangGraph library. Each node in the graph represents a specialized agent, and a shared `AgentState` object is passed between them, allowing agents to build upon each other's work.

The workflow is sequential and deterministic:

```
[User Query] -> [1. Research Manager] -> [2. Search Specialist] -> [3. Financial Analyst] -> [Final Report]
```

### The Agent Team

1.  **Research Manager**: This agent acts as the team lead. It receives the user's query and creates a structured, machine-readable `research_plan`. This plan includes a list of concise search queries and any identified stock tickers, providing clear instructions for the next agent.

2.  **Search Specialist**: This agent is the data gatherer. It executes the `research_plan` by systematically using its tools (e.g., Tavily web search) to find relevant articles and information, which it then ingests into a central ChromaDB vector store.

3.  **Financial Analyst**: This agent is the final report writer. In a critical design choice for reliability, this agent programmatically calls its tools first to retrieve all necessary context from the vector database and stock market APIs. Only after all information has been gathered is the complete context passed to the LLM for the final synthesis step, ensuring a factually grounded report.

## Key Features

- **Multi-Agent Collaboration**: Utilizes a team of three distinct agents with specialized roles and tools.
- **Stateful Orchestration**: Employs LangGraph to manage a predictable, graph-based workflow where state is explicitly passed between agents.
- **Extensive Tool Integration**: Integrates multiple external tools for enhanced capabilities, including:
  - Real-time web search (Tavily)
  - Live stock market data (yfinance)
  - A persistent vector database for shared memory (ChromaDB)
- **Modular and Extensible**: The project's organization into distinct `agents` and `tools` directories makes it straightforward to add new capabilities or agents to the team.
- **Reliable by Design**: The workflow enforces a deterministic process, particularly in the final analysis stage, to prevent common LLM failure modes like "laziness" or hallucination.

## Technical Stack

- **Orchestration**: LangGraph
- **LLM Framework**: LangChain
- **LLM Provider**: Google Gemini (e.g., `gemini-1.5-flash`)
- **Vector Database**: ChromaDB
- **Embedding Model**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Core Tools**: Tavily API, yfinance
- **Language**: Python 3.10+

## Project Structure

```
synapse-agent/
├── .env
├── requirements.txt
├── config.py
├── graph.py
├── main.py
├── agents/
│   ├── __init__.py
│   ├── research_manager.py
│   ├── search_specialist.py
│   └── financial_analyst.py
└── tools/
    ├── __init__.py
    ├── common.py
    ├── stock_tool.py
    ├── tavily.py
    └── vector_search.py
```

## Setup and Installation

Follow these steps to set up and run the project locally.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/polymathLTE/agentic-ai.git
    cd agentic-ai/synapse-agent
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r ../requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the project root by copying the example template.
    ```bash
    cp .env.example .env
    ```
    Open the `.env` file with a text editor and add your API keys:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY_HERE"
    ```

## Running the Application

To start the multi-agent system, run the `main.py` script from the project's root directory:

```bash
python3 main.py
```

The application will initialize and present you with an interactive command prompt where you can enter your queries.

## Example Usage

```
Initializing Multi-Agent Financial Team...
Agent Team Ready. Ask a financial or policy question.
Type 'exit' or 'quit' to end.

➜ How has NVIDIA's (NVDA) latest earnings call affected its stock price and recent news coverage?
```

The system will then execute the multi-agent workflow, providing status updates from each agent, and conclude by printing a comprehensive, synthesized final report.

## Future Work

- **Human-in-the-Loop**: Introduce a validation step for a human to approve or modify the `research_plan` before execution.
- **Agent Expansion**: Add new specialized agents, such as a Data Visualization Agent for generating charts or a Policy Analyst for interpreting regulatory documents.
- **Formal Evaluation**: Implement a benchmarking suite to formally evaluate the system's performance against other models on a standardized set of complex queries.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
