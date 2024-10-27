from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import os
from tempfile import TemporaryDirectory

# Initialize temporary directory for file operations
working_directory = TemporaryDirectory()

# Initialize the LLM
llm = ChatOllama(model="llama3.1")

# Define the prompt template
template = """
You are an AI assistant that can perform various calculations based on user input regarding shapes. 
User's request: {question}

Memory: {{memory}}

{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template)

# Using ConversationBufferMemory to handle conversation history
memory = ConversationBufferMemory(memory_key="memory")

# Tool for calculating volume of a cylinder
@tool
def volume_cylinder(radius: float, height: float) -> float:
    """Calculates the volume of a cylinder."""
    return 3.14159 * (radius ** 2) * height

# Tool for calculating area of a circle
@tool
def area_circle(radius: float) -> float:
    """Calculates the area of a circle."""
    return 3.14159 * (radius ** 2)

# Tool for calculating circumference of a circle
@tool
def circumference_circle(radius: float) -> float:
    """Calculates the circumference of a circle."""
    return 2 * 3.14159 * radius

# Tool for calculating volume of a rectangular box
@tool
def volume_box(length: float, width: float, height: float) -> float:
    """Calculates the volume of a rectangular box."""
    return length * width * height

# Tool for calculating surface area of a rectangular box
@tool
def surface_area_box(length: float, width: float, height: float) -> float:
    """Calculates the surface area of a rectangular box."""
    return 2 * (length * width + width * height + length * height)

# Tool for calculating volume of a sphere
@tool
def volume_sphere(radius: float) -> float:
    """Calculates the volume of a sphere."""
    return (4/3) * 3.14159 * (radius ** 3)

# Tool for writing to a file
@tool
def write_file(file_path: str, output) -> str:
    """Writes output to a specified file after converting it to a string."""
    full_path = os.path.join(working_directory.name, file_path)
    with open(full_path, 'w') as f:
        f.write(str(output))  # Convert output to string
    return f"File '{file_path}' written successfully!"

# Combine all tools
tools = [
    volume_cylinder, 
    area_circle, 
    circumference_circle, 
    volume_box, 
    surface_area_box, 
    volume_sphere, 
    write_file
]

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools)

def main():
    # Single user query specifying multiple shapes
    user_query = input("You:")

    try:
        # Invoke the agent with a single query
        output = agent_executor.invoke({"question": user_query})
        print("Agent's response:")
        print(output['output'])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
