import time
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder

from retrieval import index_files
from agent import query_pipeline, run_pipeline  # This is the async generator from your agent code

app = FastAPI()


class OpenAIQuery(BaseModel):
    model: str
    messages: list
    stream: bool
    temperature: float = None


@app.post("/v1/chat/completions")
async def chat_completions_stream(query: OpenAIQuery):

    if not query.stream:
        reply = await run_pipeline(query.messages)

        response = {
            "id": "chatcmpl-AXXyzrd626obzrJ02HBl9LXS3AJnp",
            "object": "chat.completion",
            "created": time.time(),
            "model": "chatgpt-4o-latest",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply,
                    },
                    "finish_reason": "stop"
                }
            ],
        }

        return response

    async def stream_generator():
        i = 0

        async for content in query_pipeline(query.messages):
            chunk = {
                "id": f"a{i}",
                "object": "chat.completion.chunk",
                "created": time.time(),
                "model": "haystack-agent",
                "choices": [
                    {
                        "delta": {
                            "content": content
                        }
                    }
                ],
            }
            yield f"data: {json.dumps(jsonable_encoder(chunk))}\n\n"
            i += 1

        # When done, send the [DONE] message
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "haystack-agent",
                "object": "model",
                "created": 1691000000,
                "owned_by": "organization"
            }
        ]
    }


@app.post("/index")
def run_indexing():
    index_files()
    return {"message": "Indexing completed"}
