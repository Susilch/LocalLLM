from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


llm = OllamaLLM(model="phi3")

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b 

@tool 
def multiplly(a: int, b: int) -> int:
    """Muliplies a and b."""
    return a * b

tools = [add, multiplly]

llm_with_tools = llm.bind_tools(tools)

question = "What is 14 * 3? Also, What is 5 + 47?"

message =[HumanMessage(content=question)]

ai_msg = llm_with_tools.invoke(message)


