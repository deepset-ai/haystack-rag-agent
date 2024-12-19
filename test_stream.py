import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from asyncio import Queue
from haystack_experimental.components.generators.chat import OpenAIChatGenerator
from haystack_experimental.core import AsyncPipeline
from haystack_experimental.components.builders import ChatPromptBuilder
from haystack_experimental.dataclasses import ChatMessage
from haystack.dataclasses import StreamingChunk
from typing import AsyncGenerator, Callable

app = FastAPI()


class ChatQuery(BaseModel):
    """
    Model representing a chat query request.

    :param query: The input text message from the user.
    """
    query: str


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


template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(template)])
generator = OpenAIChatGenerator(model="gpt-4o-mini")

# Create pipeline
async_rag_pipeline = AsyncPipeline()
async_rag_pipeline.add_component("prompt_builder", prompt_builder)
async_rag_pipeline.add_component("llm", generator)
async_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")


async def query_pipeline(question: str) -> AsyncGenerator[str, None]:
    """
    Query the pipeline with a question and stream the response.

    :param question: The question to ask
    :returns: AsyncGenerator yielding response chunks
    """
    request_collector = ChunkCollector()

    # Create an async callback
    async def callback(chunk):
        await collect_chunk(request_collector.queue, chunk)

    input_data = {
        "prompt_builder": {"question": question, "documents": []},
        "llm": {"streaming_callback": callback}
    }

    async def pipeline_runner():
        async for _ in async_rag_pipeline.run(input_data):
            pass
        await request_collector.queue.put(None)

    asyncio.create_task(pipeline_runner())
    async for chunk in request_collector.generator():
        yield chunk


@app.post("/chat")
async def chat(chat_query: ChatQuery):
    """
    Process a chat query and return a streaming response.

    This endpoint handles incoming chat messages and returns streamed responses
    from the Haystack pipeline.

    :param chat_query: The chat query containing the user's message
    :returns: StreamingResponse containing the generated response
    """
    return StreamingResponse(
        query_pipeline(chat_query.query),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)