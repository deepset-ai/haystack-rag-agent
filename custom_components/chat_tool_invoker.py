from typing import Any, Dict, List
from haystack import component
from haystack_experimental.dataclasses import ChatMessage
from haystack_experimental.components.tools import ToolInvoker

@component
class ChatToolInvoker(ToolInvoker):
    def __init__(self, **kwargs):
        super(ChatToolInvoker, self).__init__(**kwargs)

    @component.output_types(tool_messages=List[ChatMessage])
    def run(self, messages: List[ChatMessage], *args, **kwargs) -> Dict[str, Any]:

        parent_result = super(ChatToolInvoker, self).run([messages[-1]], *args, **kwargs)

        tool_messages = parent_result["tool_messages"]
        combined_messages = messages + tool_messages

        return {"tool_messages": combined_messages}
