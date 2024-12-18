from typing import Any, Dict, List, Optional
from haystack import component
from haystack_experimental.dataclasses import ChatMessage, Tool, ChatRole
import json

@component
class AgentVisualizer:
    def __init__(self, tools: Optional[List[Tool]] = None):
        self.tools = tools or []

    def run(self, messages: List[ChatMessage], tools: Optional[List[Tool]] = None) -> Dict[str, Any]:
        tool_names = [tool.name for tool in tools] if tools else []
        tool_calls = self.extract_tool_calls(messages)

        if not tool_calls:
            return {"output": messages[-1].text if messages else "No messages available"}

        visualization = self.visualize_toolcalls(tool_names, tool_calls)

        last_message_text = messages[-1].text if messages else "No messages available"

        output = f"{visualization}\n---\n{last_message_text}"

        return {"output": output}

    def extract_tool_calls(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        tool_calls = []

        for message in messages:
            # Check if the message contains a ToolCallResult
            if message.role == ChatRole.TOOL and message.tool_call_result:
                result = message.tool_call_result
                tool_calls.append({
                    "tool_name": result.origin.tool_name,
                    "parameters": result.origin.arguments,
                    "result": result.result,
                    "error": result.error
                })

        return tool_calls

    def visualize_toolcalls(self, tool_names: List[str], executed_tools: List[Dict[str, Any]]) -> str:
        # Create Mermaid graph
        mermaid_graph = [
            "```mermaid",
            "graph TD;",
            "    B([Decision]);"
        ]

        executed_tool_names = {et["tool_name"] for et in executed_tools}

        for i, tool_name in enumerate(tool_names, start=1):
            if tool_name in executed_tool_names:
                mermaid_graph.append(f"    B --> T{i}([{tool_name}]);")
            else:
                mermaid_graph.append(f"    B -.-> T{i}([{tool_name}]):::greyedOut;")

        mermaid_graph.append("")
        mermaid_graph.append("    %% Define the class for greyed-out nodes")
        mermaid_graph.append("    classDef greyedOut fill:#d3d3d3,stroke:#a9a9a9,color:#696969;")
        mermaid_graph.append("```")

        # Create Markdown formatted tool call details
        markdown_output_lines = []
        for executed_tool in executed_tools:
            tool_name = executed_tool["tool_name"]
            parameters = executed_tool["parameters"]
            result = executed_tool["result"]

            tool_block = [
                f"> **`{tool_name}`**",
                ">",
                "> |        |               |",
                "> |--------|---------------|"
            ]

            for param, value in parameters.items():
                tool_block.append(f"> | {param.capitalize()} | **{value}** |")

            tool_block.append(">")  # Blank line before blockquote

            # Handle multiline result by prefixing each line with '>'
            if isinstance(result, str):
                multiline_result = "\n".join(f"> {line}" for line in result.splitlines())
                tool_block.append(multiline_result)
            else:
                tool_block.append(f"> {result}")

            markdown_output_lines.append("\n".join(tool_block))
            markdown_output_lines.append("")  # Extra newline between tools

        # Join all tool blocks with double newlines to separate them
        markdown_string = "\n\n".join(markdown_output_lines)

        # Combine Mermaid graph and Markdown tool details
        mermaid_string = "\n".join(mermaid_graph)
        return f"{mermaid_string}\n\n{markdown_string}"
