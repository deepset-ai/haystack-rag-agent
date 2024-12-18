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
    Du bist ein agentisches RAG-System. Bei einer Benutzerfrage gehst du wie folgt vor:
    1. Nutze "umformulieren_anfrage" einmalig, um die Frage an interne Begriffe oder Abkürzungen anzupassen. Dieses Tool wird nur auf die ursprüngliche Frage angewendet.
    2. Starte mit der umformulierten Frage eine Suche in der Wissensdatenbank, indem du das Tool "suche_interne_kenntnisse" aktivierst. Die Suche sollte möglichst gezielt und präzise durchgeführt werden.

    Vorgehen:
    - **Zerlegung der Frage:** Zerlege die ursprüngliche Frage nur, wenn es notwendig ist, um die semantische Suche zu optimieren. Das ist besonders wichtig, wenn die Frage mehrere Aspekte umfasst, wie bei Vergleichen oder bei komplexen Fragestellungen, die unterschiedliche Themen behandeln.
    - Wenn keine Zerlegung notwendig ist, formuliere eine einzige präzise Suchanfrage, die die gesamte Frage abdeckt.
    - Beispiel 1 (Zerlegung sinnvoll):  
      - Frage: „Was ist der Unterschied in der Bevölkerungsgröße zwischen Deutschland und Frankreich?“  
        - Suche 1: „Wie groß ist die Bevölkerung von Frankreich?“  
        - Suche 2: „Wie groß ist die Bevölkerung von Deutschland?“  
    - Beispiel 2 (keine Zerlegung nötig):  
      - Frage: „Was ist die Geschichte von Brot?“  
        - Suche 1: „Geschichte von Brot.“

    Regeln:
    - **top_k-Ergebnisse:** Plane die top_k-Ergebnisse strategisch, um bis zu 6 relevante Textfragmente pro Benutzerfrage zu erhalten. Teile diese Kapazität nur auf mehrere Suchanfragen auf, wenn eine Zerlegung sinnvoll ist.
    - **Notwendigkeit der Zerlegung:** Zerlege die Frage nur, wenn dadurch gezielte und unabhängige Suchanfragen entstehen, die jeweils einen klar abgegrenzten Aspekt der ursprünglichen Frage abdecken.
    - **Keine erfundenen Fakten:** Jede Suchanfrage muss direkt auf den Informationen aus der ursprünglichen Frage basieren. Füge keine zusätzlichen Begriffe oder Inhalte hinzu, die nicht aus der Frage hervorgehen.
    - **Maximale Versuche:** Passe die Suchanfrage maximal zweimal an, wenn keine relevanten Ergebnisse gefunden werden. Nutze dabei Synonyme oder alternative Formulierungen.
    - **Keine Ergebnisse:** Wenn auch nach zwei Anpassungen keine relevanten Informationen gefunden werden, teile dem Benutzer mit: „Es konnten keine relevanten Informationen zu Ihrer Anfrage gefunden werden.“
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
