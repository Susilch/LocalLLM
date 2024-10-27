from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
import os
from fpdf import FPDF
import csv
import math
import uuid  # For generating unique file names

# Initialize the AI model
llm = ChatOllama(model="llama3.1")

# Directory to save output files
output_directory = "/home/susil/Documents/LocalLLM"

# Prompt template for the agent
template = """
You are an AI assistant with access to various mathematical and geometric tools.
Analyze the given question and choose the appropriate tool to solve it. and
print the results in files format specified.
If the question cannot be answered with these tools, say so.

Include your intermediate reasoning in the scratchpad.

Question: {question}

Memory: {{memory}}

{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template)

memory = []

@tool 
def volume_box(length: float, width: float, height: float) -> str:
    """Calculate the volume of a rectangular box."""
    return str(length * width * height)

@tool 
def volume_sphere(radius: float) -> str:
    """Calculate the volume of a sphere."""
    return str((4/3) * math.pi * radius**3)

@tool 
def volume_cylinder(radius: float, height: float) -> str:
    """Calculate the volume of a cylinder."""
    return str(math.pi * radius**2 * height)

@tool 
def surface_area_box(length: float, width: float, height: float) -> str:
    """Calculate the surface area of a rectangular box."""
    return str(2 * (length * width + width * height + height * length))

@tool 
def surface_area_sphere(radius: float) -> str:
    """Calculate the surface area of a sphere."""
    return str(4 * math.pi * radius**2)

@tool 
def surface_area_cylinder(radius: float, height: float) -> str:
    """Calculate the surface area of a cylinder."""
    return str(2 * math.pi * radius * (radius + height))

@tool 
def area_rectangle(length: float, width: float) -> str:
    """Calculate the area of a rectangle."""
    return str(length * width)

@tool 
def area_triangle(base: float, height: float) -> str:
    """Calculate the area of a triangle."""
    return str(0.5 * base * height)

@tool 
def circumference_circle(radius: float) -> str:
    """Calculate the circumference of a circle."""
    return str(2 * math.pi * radius)

@tool 
def area_circle(radius: float) -> str:
    """Calculate the area of a circle."""
    return str(math.pi * radius**2)

@tool 
def create_file(text: str, format_type: str) -> str:
    """Create a file in the specified format (PDF, TXT, or CSV).
    
    Args:
        text (str): The text to include in the file.
        format_type (str): The format of the file (PDF, TXT, or CSV).

    Returns:
        str: Confirmation message indicating success or failure.
    """
    # Generate a unique file name
    file_name = f"output_{uuid.uuid4()}.{format_type.lower()}"
    
    if format_type.lower() == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        full_path = os.path.join(output_directory, file_name)
        pdf.output(full_path)
        return f"PDF written successfully to {file_name}!"

    elif format_type.lower() == 'txt':
        with open(os.path.join(output_directory, file_name), 'w') as f:
            f.write(text)
        return f"TXT written successfully to {file_name}!"

    elif format_type.lower() == 'csv':
        with open(os.path.join(output_directory, file_name), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Result"])
            writer.writerow([text])
        return f"CSV written successfully to {file_name}!"

    return "Error: Unsupported format type. Please use 'PDF', 'TXT', or 'CSV'."

# Register the tools
tools = [
    volume_box,
    volume_sphere,
    volume_cylinder,
    surface_area_box,
    surface_area_sphere,
    surface_area_cylinder,
    area_rectangle,
    area_triangle,
    circumference_circle,
    area_circle,
    create_file
]

# Create the tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools)

def main():
    while True:
        question = input("You (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        try:
            # Invoke the agent with the question
            output = agent_executor.invoke({"question": question})
            print("Agent's response:")
            print(output['output'])

            # Store the question and answer in memory
            memory.append({"question": question, "answer": output['output']})

            # Extract the agent's scratchpad (intermediate thoughts)
            agent_scratchpad = output.get("intermediate_steps", "")
            text_output = f"Result: {output['output']}\n\nAgent Scratchpad:\n{agent_scratchpad}"

            # Automatically determine format type (e.g., based on the content)
            format_type = 'txt'  # Change this as needed based on logic or model response
            file_status = create_file.invoke({"text": text_output, "format_type": format_type})
            print(file_status)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
