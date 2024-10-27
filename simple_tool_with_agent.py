from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
import os
from tempfile import TemporaryDirectory



# Ensure you're using the correct model name
llm = ChatOllama(model="llama3.1")
  
template = """
You are an AI assistant with access to two mathematical tools: addition and multiplication.
Analyze the given question and choose the appropriate tool to solve it.
If the question cannot be answered with these tools, say so.

and the output is written in the file.

Question: {question}

Memory: {{memory}}

{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template)

memory = []


@tool 
def add(a: float, b: float) -> str:
    """Adds two numbers a and b."""
    return "added"


@tool 
def multiply(a: float, b: float) -> str:
    """Multiply two numbers a and b."""
    return "multiplied"

@tool("Create_file_tool")
def Create_file(file_path: str, text: str) ->str:
    """This will create file of the output generated to user question"""
    with TemporaryDirectory() as working_directory:
     full_path = os.path.join(working_directory, file_path)
    with open(full_path,'w') as f:
        f.write(text)
    return f"File written sucessfully!"



tools = [add, multiply, Create_file]

agent = create_tool_calling_agent(llm, tools, prompt)

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