from typing import Annotated
from haystack_experimental.dataclasses import Tool


def spawn_car(
    car_id: Annotated[str, "Unique identifier for the car"] = "Car1",
    model: Annotated[str, "The model of the car"] = "Sedan"):
    """Simulates spawning a car instance in the simulation."""
    return f"Car '{car_id}' of model '{model}' has been spawned."


def delete_car(
    car_id: Annotated[str, "Unique identifier for the car to delete"] = "Car1"):
    """Simulates deleting a car instance in the simulation."""
    return f"Car '{car_id}' has been deleted from the simulation."


def get_car_simulation_tools():
    tools = []
    for name, obj in globals().items():
        if callable(obj) and obj.__module__ == __name__ and name != "get_car_simulation_tools":
            tools.append(Tool.from_function(obj))
    return tools
