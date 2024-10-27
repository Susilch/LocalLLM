from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

llm = ChatOllama(model="llama3.1")


@tool 
def add(a: int, b: int) -> int:
    """Adds two numbers a and b."""
    return a + b


@tool 
def multiply(a: int, b: int) -> int:
    """ Multiply two numbers a and b."""
    return a * b


tools = [add, multiply]

llm_with_tools = llm.bind_tools(tools)

question = "What is 3 *12?"

messages = [HumanMessage(question)]

ai_msg = llm_with_tools.invoke(messages)

print(ai_msg.tool_calls)

messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)


messages
print(llm_with_tools.invoke(messages))

