import time

from fastapi import FastAPI
from pydantic import BaseModel

from agent import run_pipeline

app = FastAPI()


class OpenAIQuery(BaseModel):
    model: str
    messages: list
    stream: bool
    temperature: float = None


@app.post("/v1/chat/completions")
def chat_completions_stream(query: OpenAIQuery):

    reply = run_pipeline(query.messages)

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
