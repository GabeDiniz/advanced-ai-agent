from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model

llm = Ollama(
    model="mistral",
    request_timeout=30,
)

result = llm.complete("Hello, how are you?")
print(result)
