import os
from dotenv import load_dotenv
from haystack_experimental.dataclasses import ChatMessage, ToolCallResult, ToolCall
from haystack_experimental.components.generators.chat import OpenAIChatGenerator
from haystack_experimental.core import AsyncPipeline
from haystack.utils import Secret
from custom_components.chat_tool_invoker import ChatToolInvoker
from custom_components.openai_agent import OpenAIAgent
from custom_components.agent_visualizer import AgentVisualizer
from tools import get_tools
from typing import AsyncGenerator

import asyncio
from asyncio import Queue
from haystack.dataclasses import StreamingChunk
from haystack_experimental.dataclasses import Tool
from typing import Annotated, Literal

from car_simulation_tools import get_car_simulation_tools  # Import car simulation tools

_pipeline = None
_tools = None
_car_simulation_agent = None
_car_simulation_tools = None

class ChunkCollector:
    """
    Collector that stores chunks in an async queue.
    """

    def __init__(self):
        self.queue = Queue()

    async def generator(self) -> AsyncGenerator[str, None]:
        """
        Generate chunks from the queue.

        :returns: AsyncGenerator yielding string chunks
        """
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                break
            yield chunk

async def collect_chunk(queue: Queue, chunk: StreamingChunk):
    """
    Collect chunks and store them in the queue.

    :param queue: Queue to store the chunks
    :param chunk: StreamingChunk to be collected
    """
    await queue.put(chunk.content)


def initialize_pipeline():
    load_dotenv()
    tools = get_tools()

    tool_invoker = ChatToolInvoker(tools=tools)
    generator = OpenAIChatGenerator(api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")), model="gpt-4o")
    llm = OpenAIAgent(generator=generator)

    # Use AsyncPipeline instead of Pipeline
    pipeline = AsyncPipeline()

    pipeline.add_component("llm", llm)
    pipeline.add_component("tool_invoker", tool_invoker)
    pipeline.add_component("agent_visualizer", AgentVisualizer())

    # Very simple agent loop
    pipeline.connect("llm.tool_reply", "tool_invoker.messages")
    pipeline.connect("tool_invoker.tool_messages", "llm.followup_messages")
    pipeline.connect("llm.chat_history", "agent_visualizer.messages")

    return pipeline, tools



def get_pipeline():
    global _pipeline, _tools
    if _pipeline is None or _tools is None:
        _pipeline, _tools = initialize_pipeline()
    return _pipeline, _tools


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


async def query_pipeline(messages) -> AsyncGenerator[str, None]:
    """
    Asynchronously query the pipeline and stream the response.
    """
    request_collector = ChunkCollector()

    pipeline, tools = get_pipeline()

    # System message
    system_message = """
        Du bist ein agentisches RAG-System. Bearbeite Benutzerfragen in 3 Schritten:
        1. **Umformulierung:** Nutze "umformulieren_anfrage" genau einmal zu Beginn, um die originale Frage des Benutzers an interne Begriffe oder Abkürzungen anzupassen.

        2. **Zerlegungsregeln (nur wenn nötig):** Zerlege die umformulierte Frage ausschließlich, wenn:
           - **Vergleich:** Ein expliziter Vergleich enthalten ist (z. B. „Was ist der Unterschied zwischen A und B?“).
           - **Mehrere Themen:** Die Frage mehrere klar abgegrenzte Themen oder Aspekte umfasst, die unabhängig voneinander beantwortet werden können.
           - **Keine Zerlegung:** Wenn die umformulierte Frage bereits präzise und semantisch sinnvoll gestellt ist (z. B. „Was ist die Geschichte von Brot?“), führe **keine Zerlegung** durch und nutze die gesamte Frage für die Suche.

           **Beispiele für Zerlegung in verschiedene Suchanfragen des Tools "suche_interne_kenntnisse":**
           - Ursprünglich: „Unterschied zwischen Zubereitung von Brot heute und früher?“
             - Teilfragen:
               1. „Wie wurde Brot früher zubereitet?“
               2. „Wie wird Brot heute zubereitet?“
           - Ursprünglich: „Wie ist die Geschichte von Brot und wie wird es heute industriell hergestellt?“
             - Teilfragen:
               1. „Was ist die Geschichte von Brot?“
               2. „Wie wird Brot heute industriell hergestellt?“

        3. **Suche:** Führe gezielte Suchanfragen mit "suche_interne_kenntnisse" auf Basis der Zerlegungsregeln durch:
           - Ergänze keine neuen Informationen oder Aspekte, die nicht explizit in der umformulierten Frage stehen.
           - Die Suchanfragen sollen präzise Teilfragen der ursprünglichen Frage sein, ohne Vermutungen oder zusätzliche Inhalte.
           - Verwende 6 relevante Textfragmente pro Benutzerfrage (top_k). Nutze diese clever aus, so dass du sie geschickt aufteilst, falls du mehrere Suchanfrage auslöst.
           - Wenn keine Ergebnisse gefunden werden, passe die Suchanfrage maximal zweimal an (Synonyme/Alternativen).
           - Informiere den Benutzer bei ausbleibenden Ergebnissen: „Es konnten keine relevanten Informationen zu Ihrer Anfrage gefunden werden.“
           - Führe also für eine zerlegte Suche iterativ einen Tool Call aus.

        **Regeln:**
        - Ergänze keine neuen Inhalte oder Details in den Suchanfragen. Die Suchanfragen basieren nur auf der ursprünglichen Frage oder deren Teilfragen.
        - Zerlege die Frage nur, wenn ein Vergleich oder mehrere klar abgegrenzte Themen vorhanden sind.
        - Vermeide redundante, generische oder irrelevante Teilfragen.
        - Keine erfundenen Fakten, nutze nur Informationen aus der Frage.
        """

    system_message = [ChatMessage.from_system(system_message)]
    messages = convert_to_chat_message_objects(messages)
    messages = system_message + messages

    async def callback(chunk: StreamingChunk):
        if isinstance(chunk.content, ChatMessage) and chunk.content._content:
            for item in chunk.content._content:
                if isinstance(item, ToolCallResult):
                    origin_call = item.origin
                    tool_name = origin_call.tool_name
                    arguments = origin_call.arguments
                    result = item.result

                    arg_table_rows = "\n".join(
                        f"  | {key} | {value} |" for key, value in arguments.items()
                    )

                    md = (
                        f"**{tool_name}**\n"
                        "<details>\n"
                        "  <summary>Click to expand</summary>\n\n"
                        "  | Parameter | Value |\n"
                        "  |-----------|-------|\n"
                        f"{arg_table_rows}\n\n"
                        f"{result}\n"
                        "</details>"
                    )

                    await request_collector.queue.put(md)
                    return

        # Falls kein ToolCallResult gefunden wurde, normaler Ablauf
        await collect_chunk(request_collector.queue, chunk)

    input_data = {
        "llm": {"messages": messages, "tools": tools, "streaming_callback": callback},
        "agent_visualizer": {"tools": tools}
    }

    async def pipeline_runner():
        async for content in pipeline.run(
                data={
                    "llm": {"messages": messages, "tools": tools, "streaming_callback": callback},
                    "agent_visualizer": {"tools": tools},
                },
        ):
            print(f"chunk: {chunk}")# pass
        await request_collector.queue.put(None)

    asyncio.create_task(pipeline_runner())
    async for chunk in request_collector.generator():
        yield chunk


async def run_pipeline(messages):
    pipeline, tools = get_pipeline()

    system_message = """
    Du bist ein agentisches RAG-System. Bearbeite Benutzerfragen in 3 Schritten:

    1. **Umformulierung:** Nutze "umformulieren_anfrage" genau einmal zu Beginn, um die originale Frage des Benutzers an interne Begriffe oder Abkürzungen anzupassen.

    2. **Zerlegung (nur wenn nötig):** Zerlege die umformulierte Frage ausschließlich, wenn:
       - **Vergleich:** Ein expliziter Vergleich enthalten ist (z. B. „Was ist der Unterschied zwischen A und B?“).
       - **Mehrere Themen:** Die Frage mehrere klar abgegrenzte Themen oder Aspekte umfasst, die unabhängig voneinander beantwortet werden können.
       - **Keine Zerlegung:** Wenn die umformulierte Frage bereits präzise und semantisch sinnvoll gestellt ist (z. B. „Was ist die Geschichte von Brot?“), führe **keine Zerlegung** durch und nutze die gesamte Frage für die Suche.

       **Beispiele für Zerlegung:**
       - Ursprünglich: „Unterschied zwischen Zubereitung von Brot heute und früher?“
         - Teilfragen:
           1. „Wie wurde Brot früher zubereitet?“
           2. „Wie wird Brot heute zubereitet?“
       - Ursprünglich: „Wie ist die Geschichte von Brot und wie wird es heute industriell hergestellt?“
         - Teilfragen:
           1. „Was ist die Geschichte von Brot?“
           2. „Wie wird Brot heute industriell hergestellt?“

    3. **Suche:** Führe gezielte Suchanfragen mit "suche_interne_kenntnisse" durch:
       - Ergänze keine neuen Informationen oder Aspekte, die nicht explizit in der umformulierten Frage stehen.
       - Die Suchanfragen sollen präzise Teilfragen der ursprünglichen Frage sein, ohne Vermutungen oder zusätzliche Inhalte.
       - Plane maximal 6 relevante Textfragmente pro Benutzerfrage.
       - Wenn keine Ergebnisse gefunden werden, passe die Suchanfrage maximal zweimal an (Synonyme/Alternativen).
       - Informiere den Benutzer bei ausbleibenden Ergebnissen: „Es konnten keine relevanten Informationen zu Ihrer Anfrage gefunden werden.“

    **Regeln:**
    - Ergänze keine neuen Inhalte oder Details in den Suchanfragen. Die Suchanfragen basieren nur auf der ursprünglichen Frage oder deren Teilfragen.
    - Zerlege die Frage nur, wenn ein Vergleich oder mehrere klar abgegrenzte Themen vorhanden sind.
    - Vermeide redundante, generische oder irrelevante Teilfragen.
    - Keine erfundenen Fakten, nutze nur Informationen aus der Frage.
    """

    system_message = [ChatMessage.from_system(system_message)]
    messages = convert_to_chat_message_objects(messages)
    messages = system_message + messages

    final_result = None
    async for result in pipeline.run(
            data={
                "llm": {"messages": messages, "tools": tools},
                "agent_visualizer": {"tools": tools},
            },
            # include_outputs_from=["llm", "tool_invoker"]
    ):
        final_result = result

    output = final_result["agent_visualizer"]["output"]
    return output