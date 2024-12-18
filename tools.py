from typing import Annotated, Literal
from haystack_experimental.dataclasses import Tool
from retrieval import run_pipeline

# def get_weather(
#     city: Annotated[str, "the city for which to get the weather"] = "Munich",
#     unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius"):
#     """A simple function to get the current weather for a location."""
#     return f"22.0 Celsius"
#
# def get_traffic_time(
#     start_city: Annotated[str, "the city from which the route starts"] = "Munich",
#     desination_city: Annotated[str, "the city where the routes ends"] = "Dusseldorf"):
#     """A simple function that returns the current traffic time between two cities."""
#     return f"4 hours"
#
# def get_crypto_price(
#     cryptocurrency: Annotated[str, "The cryptocurrency to check (e.g., 'Bitcoin', 'Ethereum')"] = "Bitcoin",
#     currency: Annotated[str, "The fiat currency to convert to (e.g., 'USD', 'EUR')"] = "USD"):
#     """Fetches the current price of a cryptocurrency in a given fiat currency."""
#     return "100 000 USD"


def rephrase_query(
    original_question: Annotated[str, "The plain original question"]):
    """Rephrases the question based on internal abbreviations and discriptions"""
    return original_question


def search_internal_knowledge(
        query: Annotated[str, "Query on which semantic search will be performed"],
        top_k: Annotated[int, "The number of semantic similar text chunks to be returned"],
):
    """Starts a retrieval system, which searches internal knowledge based on semantic similarity to the input query."""
    result = run_pipeline(query, top_k)
    return result


def get_tools():
    tools = []
    for name, obj in globals().items():
        if callable(obj) and obj.__module__ == __name__ and name != "get_tools":
            tools.append(Tool.from_function(obj))
    return tools
