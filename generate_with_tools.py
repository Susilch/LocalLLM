from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
import os
from fpdf import FPDF

LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="generate_with_tools"

# Ensure you're using the correct model name
llm = ChatOllama(model="llama3.1")



pdf_directory = "/home/susil/Documents/LocalLLM"

template = """
You are an AI assistant with access to two mathematical tools: addition and multiplication.
Analyze the given question and choose the appropriate tool to solve it.
If the question cannot be answered with these tools, say so.

Include your intermediate reasoning in the scratchpad.

Question: {question}

Memory: {{memory}}

{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template)

memory = []

@tool 
def add(a: float, b: float) -> str:
    """Adds two numbers a and b."""
    return str(a + b)

@tool 
def multiply(a: float, b: float) -> str:
    """Multiply two numbers a and b."""
    return str(a * b)

@tool("Create_PDF_tool")
def create_pdf(file_name: str, text: str, scratchpad: str) -> str:
    """This will create a PDF of the output generated from the user's question, including the agent's scratchpad."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)

        pdf.ln(10)  # Add some space between result and scratchpad
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(0, 10, "Agent Scratchpad:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, scratchpad)  # Add the agent's scratchpad

        # Save PDF to the specified directory
        full_path = os.path.join(pdf_directory, file_name)
        pdf.output(full_path)

        return f"PDF written successfully to {file_name}!"
    
    except Exception as e:
        return f"Error creating PDF: {e}"

# Register the tools
tools = [add, multiply, create_pdf]

# Create the tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools)

def main():
    question = input("You: ")

    try:
        # Invoke the agent with the question
        output = agent_executor.invoke({"question": question})
        print("Agent's response:")
        print(output['output'])

        # Store the question and answer in memory
        memory.append({"question": question, "answer": output['output']})

        # Extract the agent's scratchpad (intermediate thoughts)
        agent_scratchpad = output.get("intermediate_steps", "")

        # Call the PDF creation tool with both result and scratchpad
        pdf_status = create_pdf.invoke({"file_name": "generateV3.pdf", "text": output['output'], "scratchpad": agent_scratchpad})
        print(pdf_status)  # Print status of PDF creation

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
