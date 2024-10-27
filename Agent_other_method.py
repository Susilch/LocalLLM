from langchain_ollama import ChatOllama
from langchain.agents import Tool, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage

def multiply(a: float, b: float) -> float:
    """Multiply two numbers a and b."""
    return float(a) * float(b)

def add(a: float, b: float) -> float:
    """Add two numbers a and b."""
    return float(a) + float(b)

# Initialize the language model
llm = ChatOllama(model="llama3.1")

# Define the tools
tools = [
    Tool(
        name="Multiply",
        func=multiply,
        description="Useful for multiplying two numbers"
    ),
    Tool(
        name="Add",
        func=add,
        description="Useful for adding two numbers"
    )
]

# Set up the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the system message
system_message = SystemMessage(content="""You are a helpful assistant that can perform mathematical operations. 
When asked to perform a calculation, always use the appropriate tool and provide the final answer clearly.
After using a tool, interpret its result and give a clear, concise answer.""")

# Create the agent
agent = OpenAIFunctionsAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Test the agent
question = "What is 3 * 12?"
response = agent_executor.run(question)
print(f"Question: {question}")
print(f"Answer: {response}")