from typing import Any, Dict, List, Optional
from haystack import component
from haystack_experimental.dataclasses import ChatMessage, Tool
# from haystack_experimental.components.generators.chat import OpenAIChatGenerator
from .openai_generator import OpenAIChatGenerator
import inspect



@component
class OpenAIAgent(OpenAIChatGenerator):
    def __init__(self, generator: Optional[OpenAIChatGenerator] = None, **kwargs):
        if generator:
            init_params = inspect.signature(OpenAIChatGenerator.__init__).parameters
            generator_kwargs = {key: value for key, value in vars(generator).items() if key in init_params}
            super(OpenAIAgent, self).__init__(**generator_kwargs)
        else:
            super(OpenAIAgent, self).__init__(**kwargs)

    @component.output_types(replies=List[ChatMessage], tool_reply=List[ChatMessage], chat_history=List[ChatMessage])
    def run(
            self,
            messages: Optional[List[ChatMessage]] = None,
            followup_messages: Optional[List[ChatMessage]] = None,
            tools: Optional[List[Tool]] = None,
            *args,
            **kwargs,
    ) -> Dict[str, Any]:


        if followup_messages:
            print(f"followup{followup_messages}")
            messages = followup_messages


        parent_result = super(OpenAIAgent, self).run(messages, tools=tools, *args, **kwargs)
        completions = parent_result["replies"]

        print(completions)

        messages.append(completions[0])

        if completions[0].tool_calls:
            return {"tool_reply": messages}

        return {"replies": completions, "chat_history": messages}

    @component.output_types(replies=List[ChatMessage], tool_reply=List[ChatMessage], chat_history=List[ChatMessage])
    async def run_async(
        self,
        messages: Optional[List[ChatMessage]] = None,
        followup_messages: Optional[List[ChatMessage]] = None,
        tools: Optional[List[Tool]] = None,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:

        if followup_messages:
            messages = followup_messages

        parent_result = await super(OpenAIAgent, self).run_async(messages, tools=tools,  *args, **kwargs)
        completions = parent_result["replies"]

        messages.append(completions[0])

        if completions[0].tool_calls:
            return {"tool_reply": messages}

        return {"replies": completions, "chat_history": messages}
