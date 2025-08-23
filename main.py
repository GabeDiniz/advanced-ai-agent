from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = Ollama(
    model="mistral",
    base_url="http://127.0.0.1:11434",
    request_timeout=60,
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

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This gives docomentation about code for an API. use this for reading docs for APIs",
        )
    )
]

code_llm = Ollama(model="codellama:latest", base_url="http://127.0.0.1:11434", request_timeout=120)
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

while (prompt := input("Enter a prompt: (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
