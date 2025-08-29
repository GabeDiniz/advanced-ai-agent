from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code_reader import code_reader
import ast

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = Ollama(
    model="mistral",
    base_url="http://127.0.0.1:11434",
    request_timeout=60,
)

# This takes our document and parses it into a vector store index
parser = LlamaParse(result_type="markdown") # can't parse python files

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
    ),
    code_reader
]

code_llm = Ollama(model="codellama:latest", base_url="http://127.0.0.1:11434", request_timeout=120)
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

"""
# Code block notes
A second LLM pass whose only job is to take a messy LLM output (code + explanation) and convert it into a clean, typed object with three fields:

description (text description),
code (just the code string),
filename (a safe filename, no special chars).

The order of operations: prompt -> agent -> messy output with explanation/code -> parser (what this is) -> clean object with three fields
"""
# Handle output parsing (formatting the output)
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_template = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_template, llm])
""""""

while (prompt := input("Enter a prompt: (q to quit): ")) != "q":
    result = agent.query(prompt)
    next_result = output_pipeline.run(response=result)
    cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))

    print(cleaned_json["code"])
    print("\n\nDescription: ", cleaned_json["description"])

    filename = cleaned_json["filename"]
