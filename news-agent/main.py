# main.py
import os
from dotenv import load_dotenv
from graph import create_agent_graph, AgentState

def main():
    """
    Main entry point for the Multi-Agent Financial Analyst.
    """
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("ERROR: Please set GOOGLE_API_KEY and TAVILY_API_KEY in your .env file.")
        return

    print("Initializing Multi-Agent Financial Team...")
    app = create_agent_graph()
    print("Agent Team Ready. Ask a financial or policy question.")
    print("Type 'exit' or 'quit' to end.")

    while True:
        try:
            question = input("\n➜  ")
            if question.lower() in ["exit", "quit"]:
                break
            
            if not question.strip():
                continue

            initial_state = {"original_query": question}
            
            # This is a more reliable way to run the graph and get the final state
            final_state = app.invoke(initial_state, {"recursion_limit": 25})

            # The final report is now directly accessible in the output
            report = final_state.get('final_report', 'No report was generated.')
            print("\nFinal Report:\n")
            print(report)

        except (KeyboardInterrupt):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
