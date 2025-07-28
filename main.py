from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model

llm = Ollama(
    model="mistral",
    request_timeout=30,
)

# This takes our document and parses it into a vector store index
parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}
# Load data from the directory "./data" using the file extractor
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# Pass these documents to the vector store index and create vector embeddings
embed_model = resolve_embed_model("local:BAAI/bge-m3") # local model
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm) # utilize the vector index as a Q&A engine

result = query_engine.query(
    "What are some of the routes in the API?"
)

print(result)
