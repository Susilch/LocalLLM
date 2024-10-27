import math
from typing import Dict, Any, List
from langchain.agents import Agent, AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function

# Define calculation functions
def calculate_volume(length: float, width: float, height: float) -> float:
    return length * width * height

def calculate_area(length: float, width: float) -> float:
    return length * width

def calculate_circumference(radius: float) -> float:
    return 2 * math.pi * radius

# Define file management functions
def save_result(filename: str, content: str) -> str:
    with open(filename, 'w') as f:
        f.write(content)
    return f"Result saved to {filename}"

def read_result(filename: str) -> str:
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"File {filename} not found"

# Define tools
def calculation_tool(calculation: str, **kwargs: Any) -> str:
    if calculation == "volume":
        return str(calculate_volume(**kwargs))
    elif calculation == "area":
        return str(calculate_area(**kwargs))
    elif calculation == "circumference":
        return str(calculate_circumference(**kwargs))
    else:
        return "Invalid calculation type"

tools = [
    Tool(
        name="Calculation",
        func=calculation_tool,
        description="Useful for performing calculations like volume, area, and circumference"
    ),
    Tool(
        name="SaveResult",
        func=save_result,
        description="Saves a result to a file"
    ),
    Tool(
        name="ReadResult",
        func=read_result,
        description="Reads a result from a file"
    )
]

# Initialize the LLM
llm = Ollama(model="llama3.1")

# Create the prompt template
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an AI assistant capable of performing calculations and managing files."),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

# Create the agent executor
memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# Main interaction loop
print("Welcome to the Calculation and File Management Assistant!")
print("You can perform calculations or manage files. Type 'exit' to quit.")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break
    
    result = agent_executor.invoke({"input": user_input})
    print("AI:", result["output"])
