from typing import Annotated, Literal
from haystack_experimental.dataclasses import Tool
from retrieval import run_pipeline


def umformulieren_anfrage(
    originalfrage: Annotated[str, "Die ursprüngliche, unveränderte Frage"]
):
    """Formuliert die Frage basierend auf internen Abkürzungen und Beschreibungen um."""
    return originalfrage


def suche_interne_kenntnisse(
    query: Annotated[str, "Eine präzise, isolierte Suchanfrage, die auf semantische Ähnlichkeit zur Eingabe basiert"],
    top_k: Annotated[int, "Die Anzahl der semantisch ähnlichen Textabschnitte, die zurückgegeben werden sollen"],
):
    """Startet ein Retrieval-System, das interne Kenntnisse durchsucht. Die Anfrage sollte spezifisch und in sich geschlossen sein, um die semantische Suche präzise auszurichten."""
    result = run_pipeline(query, top_k)
    return result



def get_tools():
    tools = []
    for name, obj in globals().items():
        if callable(obj) and obj.__module__ == __name__ and name != "get_tools":
            tools.append(Tool.from_function(obj))
    return tools
