from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool 
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 20 degrees Celsius and foggy."
    else: 
        return "It's 30 degrees and sunny."


llm = ChatOllama(model="llama3.1").bind_tools([get_weather])

# invoke the model
ai_msg = llm.invoke("What's the weather in sf?")


print(ai_msg.tool_calls)  #show the tool call results
