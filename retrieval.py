import os
from pathlib import Path
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.utils import Secret
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage


USER_MESSAGE_TEMPLATE = """
Dokumente:
{% for document in documents %}
Document[{{ loop.index }}]
{{ document.content }}
{% endfor %}
"""

# Load environment variables
load_dotenv()

def init_document_store():
    return OpenSearchDocumentStore(
        hosts=os.getenv("OPENSEARCH_HOST", "http://opensearch:9200"),
        username=os.getenv("OPENSEARCH_USERNAME", "admin"),
        password=os.getenv("OPENSEARCH_PASSWORD", "XYZ_123"),
        index="document",
        embedding_dim=768,
        similarity="cosine",
    )


def create_pipeline():
    document_store = init_document_store()
    pipeline = Pipeline()

    # Add retriever component
    retriever = OpenSearchBM25Retriever(document_store=document_store, top_k=5)
    chat_prompt_builder = ChatPromptBuilder(template=[
        ChatMessage.from_user(USER_MESSAGE_TEMPLATE)
    ])
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("chat_prompt_builder", chat_prompt_builder)

    pipeline.connect("retriever.documents", "chat_prompt_builder.documents")

    return pipeline

def run_pipeline(query: str, top_k: int):
    pipeline = create_pipeline()

    # Run retriever
    result = pipeline.run(data={"retriever": {"query": query, "top_k": top_k}})

    return result['chat_prompt_builder']['prompt'][0].text

def init_indexing_pipeline():
    document_store = init_document_store()
    indexing_pipeline = Pipeline()

    # Add components for preprocessing and indexing
    components = [
        ("file_type_router", FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])),
        ("pypdf_converter", PyPDFToDocument()),
        ("document_joiner", DocumentJoiner()),
        ("document_cleaner", DocumentCleaner()),
        ("document_splitter", DocumentSplitter(split_by="word", split_length=250, split_overlap=50)),
        ("document_writer", DocumentWriter(document_store, policy=DuplicatePolicy.OVERWRITE)),
    ]

    for name, component in components:
        indexing_pipeline.add_component(name, component)

    # Connect components
    indexing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    indexing_pipeline.connect("pypdf_converter", "document_joiner")
    indexing_pipeline.connect("document_joiner", "document_cleaner")
    indexing_pipeline.connect("document_cleaner", "document_splitter")
    indexing_pipeline.connect("document_splitter", "document_writer")

    return indexing_pipeline


def index_files():
    indexing_pipeline = init_indexing_pipeline()

    input_dir = Path("utils/data")
    if not input_dir.exists():
        raise FileNotFoundError("Input directory does not exist. Please provide a valid path.")

    # Run the indexing pipeline
    indexing_pipeline.run(
        {"file_type_router": {"sources": list(input_dir.glob("**/*"))}}
    )

if __name__ == "__main__":


    query = "Was sind die Hauptunterschiede zwischen ChatGPT und GPT-4?"
    result = run_pipeline(query)

    print(result)
