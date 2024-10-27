import os
from langchain_ollama import ChatOllama
from tempfile import TemporaryDirectory
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Ensure you're using the correct model name
llm = ChatOllama(model="llama3.1")
  
template = """
You are an AI assistant which have acess to different tools to perform operation specified by user
in the question. You need to create the file in format specified by the user.
Question: {question}

Memory: {memory}

{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key="memory")

working_directory = TemporaryDirectory()

@tool("Create_file_tool")
def Create_file(file_path: str, text) ->str:
    full_path = os.path.join(working_directory.name, file_path)
    with open(full_path,'w') as f:
        f.write(str(text))
    return f"File written sucessfully!"



@tool 
def add(a: float, b: float) -> float:
    """Adds two numbers a and b."""
    return float(a) + float(b)


@tool 
def multiply(a: float, b: float) -> float:
    """Multiply two numbers a and b."""
    return float(a) * float(b)


tools = [add, multiply]

agent = create_tool_calling_agent(llm, tools, prompt, memory=memory)

agent_executor = AgentExecutor(agent=agent, tools=tools)


def main():

    question = "What is 3 * 12?"

    try:
        output = agent_executor.invoke({"question": question})
        print("Agent's response:")
        print(output['output'])

        memory.append({"question": question, "answer": output['output']})

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()