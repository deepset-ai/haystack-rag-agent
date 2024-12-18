from haystack_experimental.dataclasses import ChatMessage
from haystack import Pipeline
from haystack.utils import Secret

from custom_components.chat_tool_invoker import ChatToolInvoker
from custom_components.openai_agent import OpenAIAgent
from custom_components.agent_visualizer import AgentVisualizer
from tools import get_tools
from haystack_experimental.components.generators.chat import OpenAIChatGenerator

import os
from dotenv import load_dotenv

from car_simulation_tools import get_car_simulation_tools  # Import car simulation tools

from haystack_experimental.dataclasses import Tool
from typing import Annotated, Literal


# def car_simulation_agent_tool(
#     task: Annotated[str, "The task for running the simulation"] = "Bitcoin"
# ):
#     """A wrapper function to invoke the car simulation agent."""
#     car_simulation_agent, tools = get_car_simulation_agent()
#     messages = [ChatMessage.from_user(task)]
#
#     result = car_simulation_agent.run(
#         data={
#             "car_simulation_llm": {"messages": messages, "tools": tools},
#             "agent_visualizer": {"tools": tools},
#         },
#         include_outputs_from=["car_simulation_llm", "agent_visualizer"],
#     )
#     return result["agent_visualizer"]["output"]



_pipeline = None
_tools = None
_car_simulation_agent = None
_car_simulation_tools = None


def initialize_pipeline():
    load_dotenv()
    tools = get_tools()

    # tools.append(
    #     Tool.from_function(car_simulation_agent_tool)
    # )

    tool_invoker = ChatToolInvoker(tools=tools)
    generator = OpenAIChatGenerator(api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")), model="gpt-4o")
    llm = OpenAIAgent(generator=generator)

    pipeline = Pipeline()

    pipeline.add_component("llm", llm)
    pipeline.add_component("tool_invoker", tool_invoker)
    pipeline.add_component("agent_visualizer", AgentVisualizer())

    # Very simple agent loop
    pipeline.connect("llm.tool_reply", "tool_invoker.messages")
    pipeline.connect("tool_invoker.tool_messages", "llm.followup_messages")

    pipeline.connect("llm.chat_history", "agent_visualizer.messages")

    return pipeline, tools


def initialize_car_simulation_agent():
    load_dotenv()
    car_simulation_tools = get_car_simulation_tools()

    tool_invoker = ChatToolInvoker(tools=car_simulation_tools)
    generator = OpenAIChatGenerator(api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")), model="gpt-4o")
    car_simulation_llm = OpenAIAgent(generator=generator)


    pipeline = Pipeline()

    pipeline.add_component("car_simulation_llm", car_simulation_llm)
    pipeline.add_component("car_simulation_tool_invoker", tool_invoker)
    pipeline.add_component("agent_visualizer", AgentVisualizer())


    # Simple agent loop for car simulation
    pipeline.connect("car_simulation_llm.tool_reply", "car_simulation_tool_invoker.messages")
    pipeline.connect("car_simulation_tool_invoker.tool_messages", "car_simulation_llm.followup_messages")

    pipeline.connect("car_simulation_llm.chat_history", "agent_visualizer.messages")

    return pipeline, car_simulation_tools


def get_pipeline():
    global _pipeline, _tools
    if _pipeline is None or _tools is None:
        _pipeline, _tools = initialize_pipeline()
    return _pipeline, _tools


def get_car_simulation_agent():
    global _car_simulation_agent, _car_simulation_tools
    if _car_simulation_agent is None or _car_simulation_tools is None:
        _car_simulation_agent, _car_simulation_tools = initialize_car_simulation_agent()
    return _car_simulation_agent, _car_simulation_tools



def run_pipeline(messages):
    pipeline, tools = get_pipeline()
    car_simulation_agent, _ = get_car_simulation_agent()

    # Add the car simulation agent as a tool
    # tools.append(car_simulation_agent)

    system_message = """
    You are an agentic RAG system. Here are the steps you follow, when you receive a question from the user:
    1. You use the tool "rephrase_question", to first have an updated version of the question which matches the internal use of abbreviations and descrptions. The rephrase tool is only used one, on the original question. 
    2. You trigger the search tool, which returns answers from the knowledge base, which are semantically similar to the query, with which you perform the tool.

     You goal is it, to smartly use these tools, to retrieve all the relevant information. You can also use the search tool multiple times. But the rephrase tool, should only be used on the original questions, and from the rephrased question the new query / queries should be derived. 
     F.e., when the users asks a questions like: What is the difference in population size between Germany and France, you trigger these two searches:
    1. What is the population size of France
    2. What is the population size of Germany
    
    Then you use the results of both searches, to formulate the answer.
    You have a capacity of 6 text chunks per user question, which you should always make use of. So allocate the top_k wisely for you searches.
    
    If the search tool does not provide the results needed to answer the quetsion, try out a search with a adjusted query. But only try it out a maximum of 2 times. After that, end the seach and tell the user that now answer was found. For the new initiation, you can again use the same top_k as before.
    Think, which slight change in formulation could potentially help to retrieve the correct documents. It should be a significant change, so not just the change of order of words. But instead, different words should be used which represent the same base question.

    """

    system_message = [ChatMessage.from_system(system_message)]
    messages = convert_to_chat_message_objects(messages)
    messages = system_message + messages

    result = pipeline.run(
        data={
            "llm": {"messages": messages, "tools": tools},
            "agent_visualizer": {"tools": tools},
        },
        include_outputs_from=["llm", "tool_invoker"]
    )
    print(result)
    return result["agent_visualizer"]["output"]


def convert_to_chat_message_objects(messages):
    chat_message_objects = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        adapted_message = {
            "_role": role,
            "_content": [{"text": content}]
        }

        chat_message = ChatMessage.from_dict(adapted_message)
        chat_message_objects.append(chat_message)
    return chat_message_objects


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "What is the difference in population size between germany and france?"}
    ]

    response = run_pipeline(messages=messages)
    print("Response:", response)
