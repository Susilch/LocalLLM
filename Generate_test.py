from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from fpdf import FPDF
import os

# Ensure you're using the correct model name
llm = ChatOllama(model="llama3.1")

# Directory where PDFs will be saved
pdf_directory = "/home/susil/Documents/LocalLLM/Outputs_pdf"

# Prompt template
template = """
You are an AI assistant with access to several mathematical tools: addition, subtraction, multiplication, division, and power.
Analyze the given question and choose the appropriate tool to solve it.
After calculating, save the result in a PDF file.

Question: {question}

Memory: {{memory}}

{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template)

memory = []


def create_pdf(file_name: str, text: str) -> str:
    """Generates a PDF containing the result text and saves it in the specified directory."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)

        # Save PDF to the specified directory
        full_path = os.path.join(pdf_directory, file_name)
        pdf.output(full_path)

        return f"PDF written successfully to {full_path}!"
    
    except Exception as e:
        return f"Error creating PDF: {e}"


@tool 
def add(a: float, b: float) -> str:
    """Adds two numbers a and b and saves the result in a PDF."""
    result = a + b
    text = f"The result of adding {a} and {b} is {result}."
    
    # Generate PDF with result
    pdf_status = create_pdf("addition_result.pdf", text)
    
    return f"{text} {pdf_status}"


@tool 
def subtract(a: float, b: float) -> str:
    """Subtracts b from a and saves the result in a PDF."""
    result = a - b
    text = f"The result of subtracting {b} from {a} is {result}."
    
    # Generate PDF with result
    pdf_status = create_pdf("subtraction_result.pdf", text)
    
    return f"{text} {pdf_status}"


@tool 
def multiply(a: float, b: float) -> str:
    """Multiplies two numbers a and b and saves the result in a PDF."""
    result = a * b
    text = f"The result of multiplying {a} and {b} is {result}."
    
    # Generate PDF with result
    pdf_status = create_pdf("multiplication_result.pdf", text)
    
    return f"{text} {pdf_status}"


@tool 
def divide(a: float, b: float) -> str:
    """Divides a by b and saves the result in a PDF."""
    if b == 0:
        return "Error: Division by zero is not allowed."
    
    result = a / b
    text = f"The result of dividing {a} by {b} is {result}."
    
    # Generate PDF with result
    pdf_status = create_pdf("division_result.pdf", text)
    
    return f"{text} {pdf_status}"


@tool
def power(a: float, b: float) -> str:
    """Raises a to the power of b and saves the result in a PDF."""
    result = a ** b
    text = f"The result of raising {a} to the power of {b} is {result}."
    
    # Generate PDF with result
    pdf_status = create_pdf("power_result.pdf", text)
    
    return f"{text} {pdf_status}"


# Register the tools
tools = [add, subtract, multiply, divide, power]

# Create the tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools)

def main():
    question = input("You:")

    try:
        # Invoke the agent with the question
        output = agent_executor.invoke({"question": question})
        print("Agent's response:")
        print(output['output'])

        # Store the question and answer in memory
        memory.append({"question": question, "answer": output['output']})

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
