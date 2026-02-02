import os
import logging
import time
from pathlib import Path
from typing import List, Union

# --- Third Party Imports ---
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# --- Datapizza AI Imports ---
from datapizza.core.models import PipelineComponent
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.clients.google import GoogleClient
from datapizza.modules.prompt import ChatPromptTemplate
from google.genai.errors import ClientError

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = (BASE_DIR / "storage" / "qdrant_db").resolve()
CACHE_DIR = (BASE_DIR / "model_cache").resolve()
COLLECTION_NAME = "my_knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ------------------------------------------------------------------------------
# 1. Helper: Local Embedder Logic
# ------------------------------------------------------------------------------
class LocalDenseEmbedder:
    def __init__(self, model_name: str, cache_dir: str = None):
        logger.info(f"🔌 Loading Local Embedder: {model_name}")
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed(self, text: Union[str, List[str]], model_name: str = None) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        return list(self.model.embed(text))

# ------------------------------------------------------------------------------
# 2. Component: Query Embedder for Pipeline
# ------------------------------------------------------------------------------
class FastEmbedQueryComponent(PipelineComponent):
    def __init__(self, embedder_client: LocalDenseEmbedder):
        super().__init__()
        self.client = embedder_client

    def _run(self, text: str) -> List[float]:
        embeddings = self.client.embed(text)
        vector = embeddings[0]
        if hasattr(vector, "tolist"):
            return vector.tolist()
        return list(vector)

# ------------------------------------------------------------------------------
# 3. The RAG Pipeline Class
# ------------------------------------------------------------------------------
class RAGService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GOOGLE_API_KEY is missing from .env")

        # --- Initialize Clients ---
        self.local_embedder = LocalDenseEmbedder(model_name=EMBEDDING_MODEL, cache_dir=str(CACHE_DIR))
        
        # Qdrant with Windows Fix
        self.vector_store_component = QdrantVectorstore(location=":memory:", collection_name=COLLECTION_NAME)
        
        if STORAGE_DIR.exists():
            self.real_qdrant_client = QdrantClient(path=str(STORAGE_DIR))
            self.vector_store_component.client = self.real_qdrant_client
            logger.info(f"✅ Connected to Local Qdrant at {STORAGE_DIR}")
        else:
            logger.warning(f"⚠️ Storage directory {STORAGE_DIR} not found! Queries will return empty.")
            self.real_qdrant_client = None

        self.llm_client = GoogleClient(
            api_key=self.api_key,
            model="gemini-2.5-flash", 
            system_prompt="Tu sei RudyAIbot, un assistente esperto per la gestione dell'oratorio. " +
            " Il tuo compito è fornire risposte complete, dettagliate ed esaustive basate sui documenti dell'oratorio. " +
            " Non essere troppo sintetico: se il contesto contiene procedure, regole dettagliate o liste di cose da fare, " + 
            " includi tutte le informazioni pertinenti nella tua risposta. Usa elenchi puntati per chiarezza. " +
            " Se la domanda riguarda la tua identità o funzione, presentati liberamente."
        )

        # --- Build Pipeline ---
        self.pipeline = DagPipeline()
        self.pipeline.add_module("embedder", FastEmbedQueryComponent(self.local_embedder))
        self.pipeline.add_module("retriever", self.vector_store_component)
        self.pipeline.add_module("prompt_template", ChatPromptTemplate(
            user_prompt_template="User Question: {{ user_prompt }}",
            retrieval_prompt_template="Context:\n{% for chunk in chunks %}- {{ chunk.text }}\n{% endfor %}\nAnswer:"
        ))
        self.pipeline.add_module("llm", self.llm_client)

        # Connect
        self.pipeline.connect("embedder", "retriever", target_key="query_vector")
        self.pipeline.connect("retriever", "prompt_template", target_key="chunks")
        self.pipeline.connect("prompt_template", "llm", target_key="memory")
        
        logger.info("🚀 RAG DagPipeline initialized successfully (Gemini 2.5 Flash).")

    def query(self, user_input: str) -> str:
        try:
            result_dict = self.pipeline.run({
                "embedder": {"text": user_input},
                "prompt_template": {"user_prompt": user_input},
                "retriever": {"collection_name": COLLECTION_NAME, "k": 5},
                "llm": {"input": user_input}
            })
            
            response = result_dict.get("llm")
            if hasattr(response, 'text'):
                return response.text
            return str(response)

        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                return "⏳ Quota exceeded. Please wait 10-20 seconds and try again."
            logger.error(f"Google API Error: {e}")
            return f"API Error: {e}"
            
        except Exception as e:
            logger.error(f"Pipeline Run Failed: {e}", exc_info=True)
            return "I'm sorry, I encountered an error."

    def close(self):
        """Explicitly close the Qdrant client to avoid shutdown errors."""
        if self.real_qdrant_client:
            self.real_qdrant_client.close()

# ------------------------------------------------------------------------------
# Test Run
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    rag = None
    try:
        rag = RAGService()
        print("\n💬 Asking Rudy...")
        
        # Simple retry logic for the test
        for i in range(3):
            answer = rag.query("Tell me something about this document")
            if "Quota exceeded" in answer:
                print("⚠️  Hit rate limit. Retrying in 15 seconds...")
                time.sleep(15)
                continue
            else:
                print(f"\n🤖 Rudy: {answer}")
                break
                
    finally:
        if rag:
            rag.close()