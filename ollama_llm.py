import ollama
import requests
import json
import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import os
import logging
from datetime import datetime
from tqdm import tqdm
import time

SYSTEM_PROMPT = """
[system prompt here]
"""

# configure logging
# change file name and location as needed
LOG_LOCATION = "path/to/file"
logging.basicConfig(filename=f"{LOG_LOCATION}/ollama_llm.log", filemode='a', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

"""
Logging functions with timestamps.
Format: [LEVEL] [TIMESTAMP] - message
"""
def info(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[INFO] [{timestamp}] - {message}")

def warn(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.warning(f"[WARN] [{timestamp}] - {message}")

def error(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.error(f"[ERROR] [{timestamp}] - {message}")

"""
A class representing an LLM with various capabilities including memory, web search, RAG, and function calling.

Properties:
- model_name: Name of the LLM to use.
- enable_memory: Whether to enable memory for chat history.
- memory_size: Maximum number of turns to keep in memory (default: 5). 
A turn is a pair of user and assistant messages. 
A higher value allows for longer conversations but consumes more memory.
- enable_web_search: Whether to enable web search capabilities.
- web_search_api: API endpoint for web search (if enabled).
- enable_rag: Whether to enable retrieval-augmented generation (RAG), a technique for enhancing model responses with external knowledge.
- enable_functions: Whether to enable function calling capabilities.
"""
class Model:
    def __init__(self, 
                 model_name: str,
                 enable_memory: bool = False,
                 memory_size: int = 5,
                 enable_web_search: bool = False,
                 web_search_api: Optional[str] = None,
                 enable_rag: bool = False,
                 enable_functions: bool = False
                 ):
        # properties
        self.model_name = model_name
        self.enable_memory = enable_memory
        self.memory_size = memory_size
        self.enable_web_search = enable_web_search
        self.web_search_api = web_search_api
        self.enable_rag = enable_rag
        self.enable_functions = enable_functions

        # chat history and client setup
        self.chat_history: List[Dict[str, str]] = []
        self.client = ollama.Client()
        # embedding setup
        self.docs: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.embedding_model = initialize_embedding_model()

        info(f"Initialized model: {self.model_name}")
        info(f"Properties:\n{self.__dict__}\n")

    """
    Adds a turn to the chat history, maintaining the memory size limit.
    """
    def add_turn(self, role: str, content: str):
        info(f"Adding turn with role: {role}")

        self.chat_history.append({"role": role, "content": content})
        if len(self.chat_history) > self.memory_size:
            self.chat_history.pop(0)
            self.chat_history.pop(0)

    """
    Searches the web using the specified API and returns a list of results.
    If web search is not enabled or the API is not set, raises a ValueError.
    """
    def search_web(self, query: str) -> List[str]:
        if not self.enable_web_search or not self.web_search_api:
            raise ValueError("Web search is not enabled or web_search_api is not set.")

        try:
            query = self._extract_search_query(query)
            query = query.strip().lower()
            info(f"Searching web with query: {query}")
            
            headers = {"User-Agent": "Mozilla/5.0 (compatible; OllamaLLM/1.0)"}
            response = requests.get(self.web_search_api, headers=headers, params={
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1
            })

            if response.status_code == 200:
                data = response.json()
                results = []

                # try to extract useful data from the response
                if data.get("AbstractText"):
                    results.append(data["AbstractText"])
                if data.get("RelatedTopics"):
                    for topic in data["RelatedTopics"][:3]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append(topic["Text"])
                
                if not results:
                    results.append("No relevant results found.")

                return results
        except Exception as e:
            warn(f"Web search failed: {e}")

    """
    Retrieves relevant documents from the RAG system based on the query.
    Uses cosine similarity to find the most similar documents to the query.
    If RAG is not enabled or no documents are available, raises a ValueError.
    """
    def rag_retrieve(self, query: str, top_k: int = 1) -> List[str]:
        if not self.enable_rag or not self.docs:
            raise ValueError("RAG is not enabled or no documents are available.")
        
        info(f"Retrieving RAG contexts for query: {query}")
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [self.docs[i] for i in top_indices]

        if not results:
            results.append("No relevant results found.")

        return results
    
    """
    Adds new documents to the model for RAG.
    Encodes the documents using the embedding model and updates the document embeddings.
    """
    def add_documents(self, new_docs: List[str]):
        info(f"Adding {len(new_docs)} new documents to the model.")

        self.docs.extend(new_docs)
        self.doc_embeddings = self.embedding_model.encode(self.docs)

    """
    Detects if the output contains a function call and returns the function call data if found.
    If no function call is detected, returns None.
    """
    def detect_function_call(self, output: str) -> Optional[dict]:
        info("Detecting function call in output.")

        # clean up the output by removing any <think> tags
        output = output.strip()
        output = re.sub(r"</?think>", "", output)
        
        # try full json parsing first
        try:
            data = json.loads(output)
            if "function_call" in data:
                info(f"Function call detected: {data['function_call']}")
                return data["function_call"]
        except json.JSONDecodeError:
            # If JSON parsing fails, try regex matching
            pass

        try:
            # captures a json object that starts with {"function_call": and ends at closing brace
            match = re.search(r'\{[\s\n]*"function_call"\s*:\s*\{.*?\}[\s\n]*\}', output, re.DOTALL)
            if match:
                snippet = match.group(0)
                data = json.loads(snippet)
                if "function_call" in data:
                    info(f"Function call detected: {data['function_call']}")
                    return data["function_call"]
        except Exception as e:
            warn(f"Function call detection failed: {e}")
        
        info("No valid function call detected.")
        return None

    """
    Runs the model with the given prompt.
    Handles web search, RAG retrieval, and function calling based on the model's properties.
    """
    def run(self, prompt: str) -> str:
        tools = []
        info(f"Running model.")

        # determine if tools are needed based on model properties
        if self.enable_functions:
            fns = self._get_llm_tools()
            if self.enable_web_search:
                tools.append(fns[0])
            if self.enable_rag:
                tools.append(fns[1])

        # Add user prompt to chat history if memory is enabled
        if self.enable_memory:
            self.add_turn("user", prompt)

        # prepare messages for the LLM by including system prompt and chat history
        if self.enable_memory:
            messages = self.chat_history[:]
            if not any(msg["role"] == "system" and msg["content"] == SYSTEM_PROMPT for msg in messages):
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

        output = ""
        print("LLM: ", end="", flush=True)

        try:
            if "deepseek" not in self.model_name:
                info("Using tools-compatible streaming for the model.")
                for chunk in self.client.chat(
                    model=self.model_name, 
                    messages=messages, 
                    tools=tools if tools else None, 
                    stream=True
                ):
                    token = chunk.get("message", {}).get("content", "")
                    print(token, end="", flush=True)
                    output += token
            else:
                info("Using simple streaming (non-tools compatible) for the model.")
                for chunk in self.client.chat(
                    model=self.model_name, 
                    messages=messages, 
                    stream=True
                ):
                    token = chunk.get("message", {}).get("content", "")
                    print(token, end="", flush=True)
                    output += token
        except Exception as e:
            error(f"Streaming failed: {e}")
            return "An error occurred during streaming."

        print()

        # handle function calls if enabled
        if self.enable_functions:
            fn = self.detect_function_call(output)
            if fn:
                name = fn["name"]
                args = fn.get("arguments", {})

                # handle rag retrieval
                if self.enable_rag and name == "rag_retrieve":
                    rag_contexts = self.rag_retrieve(args.get("query"), top_k=args.get("top_k", 1))
                    if rag_contexts:
                        self.add_turn("system", f"Context: {' '.join(rag_contexts)}")
                # handle web search
                elif self.enable_web_search and name == "web_search":
                    web_results = self.search_web(args.get("query"))
                    if web_results:
                        self.add_turn("system", f"Web results: {web_results}")
                
                try:
                    result = self._handle_function_call(name, args)
                    result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                    self.add_turn("system", f"Tool result: {result_str}")
                except Exception as e:
                    error(f"Function call execution failed: {e}")
                    self.add_turn("system", f"Tool execution failed: {e}")
                # have the model continue with the tool result
                return self.run("Based on the tool result, continue your previous response.")

        # if no function calls, just return the output
        if self.enable_memory:
            self.add_turn("assistant", output)

        return output
    
    """
    Resets the model state, clearing chat history and documents.
    """
    def reset(self):
        self.chat_history = []
        self.docs = []
        self.doc_embeddings = None
        self.__init__(
            model_name=self.model_name,
            enable_memory=self.enable_memory,
            memory_size=self.memory_size,
            enable_web_search=self.enable_web_search,
            web_search_api=self.web_search_api,
            enable_rag=self.enable_rag,
            enable_functions=self.enable_functions
        )

        print("Model state has been reset.")

    """
    Handles function calls based on the function name and arguments.
    Currently supports web search and RAG retrieval.
    """
    def _handle_function_call(self, name: str, args: dict) -> str:
        if name == "web_search" and "query" in args:
            return self.search_web(args["query"])
        elif name == "rag_retrieve" and "query" in args:
            top_k = args.get("top_k", 1)
            return self.rag_retrieve(args["query"], top_k)
        return f"Unknown function: {name} with args: {args}"
    
    """
    Uses regex or the LLM to extract the actual web search query from a natural prompt.
    """
    def _extract_search_query(self, full_prompt: str) -> str:
        # try regex first for simple cases
        prompt = full_prompt.strip()
        match = re.search(r"^(?:search(?: the web)? for|look up|find)\s+(.*)", prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            # if regex fails, use the LLM to extract the query
            system_prompt = (
                "Answer directly and concisely. Do NOT include internal reasoning or '<think>' tags."
                "Extract the core web search query from the user's message. "
                "Respond ONLY with the search phrase. Do not add explanations or greetings."
            )
            user_prompt = f"User asked: {full_prompt}"

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return response['message']['content'].strip()
    
    """
    Retrieves the tools available for function calling.
    Returns a list of dictionaries representing the tools.
    Each tool has a name, description, and parameters.
    """
    def _get_llm_tools(self) -> List[Dict[str, str]]:
        info("Retrieving LLM tools for function calling.")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to use."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rag_retrieve",
                    "description": "Retrieve relevant documents from the RAG system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to retrieve relevant documents."
                            },
                            "top_k": {
                                "type": "integer",
                                "default": 1,
                                "description": "Number of top documents to retrieve."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        return tools

"""
Initializes the embedding model for document encoding.
Also shows a progress bar during the loading process.
"""
def initialize_embedding_model():
    with tqdm(total=100, desc="Loading embedding model") as pbar:
        pbar.update(10)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        pbar.update(90)
    return model

"""
Loads text documents from a specified folder path (for use with RAG).
Supports filtering by file extensions (default: .txt, .md).
"""
def load_text_docs(folder_path: str, extensions=None) -> list:
    if extensions is None:
        extensions = ['.txt', '.md']

    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        docs.append(content)
                except Exception as e:
                    print(f"Failed to read {full_path}: {e}")
    return docs