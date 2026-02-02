import os
import sys
import logging
import yaml
from pathlib import Path
from typing import List, Union

# Datapizza Imports
from datapizza.pipeline import IngestionPipeline
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.embedders import ChunkEmbedder
from datapizza.vectorstores.qdrant import QdrantVectorstore

# FastEmbed & Qdrant Imports
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ------------------------------------------------------------------------------
# Logging & Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Resolve Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "ingestion_config.yaml"

# ------------------------------------------------------------------------------
# Custom Wrapper for Dense Embeddings
# ------------------------------------------------------------------------------
class LocalDenseEmbedder:
    """
    Wrapper for FastEmbed's TextEmbedding (Dense) to make it compatible 
    with Datapizza's ChunkEmbedder.
    """
    def __init__(self, model_name: str, cache_dir: str = None):
        self.model = TextEmbedding(
            model_name=model_name, 
            cache_dir=cache_dir
        )

    # Accepts 'model_name' to prevent crashes, but ignores it
    def embed(self, text: Union[str, List[str]], model_name: str = None) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        return list(self.model.embed(text))

# ------------------------------------------------------------------------------
# Main Ingestion Logic
# ------------------------------------------------------------------------------
def load_config(path):
    if not path.exists():
        logger.critical(f"❌ Config file not found at: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_index():
    # 1. Load Configuration
    logger.info(f"⚙️  Loading configuration from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    
    # Extract params
    paths = config.get("paths", {})
    model_name = config.get("embedding_model", "BAAI/bge-small-en-v1.5")
    chunk_size = config.get("chunk_size", 512)
    collection_name = paths.get("collection_name", "my_knowledge_base")
    
    # Paths
    data_dir = (BASE_DIR / paths.get("data_dir", "./documents")).resolve()
    storage_dir = (BASE_DIR / paths.get("storage_dir", "./storage/qdrant_db")).resolve()
    cache_dir = (BASE_DIR / paths.get("cache_dir", "./model_cache")).resolve()
    
    # 2. Initialize Components
    try:
        logger.info("🔧 Initializing Pipeline Components...")

        # A. Parser
        parser = DoclingParser()

        # B. Splitter
        splitter = NodeSplitter(max_char=chunk_size)

        # C. Embedder
        logger.info(f"   - Loading Dense Embedder: {model_name}")
        local_client = LocalDenseEmbedder(
            model_name=model_name,
            cache_dir=str(cache_dir)
        )
        embedder = ChunkEmbedder(client=local_client)

        # D. Vector Store (THE FIX)
        logger.info(f"   - Initializing Qdrant Storage at: {storage_dir}")
        
        # Step D1: Init wrapper in Memory mode to pass validation checks
        vector_store = QdrantVectorstore(
            location=":memory:", 
            collection_name=collection_name
        )
        
        # Step D2: SWAP the client manually to force Local Disk Mode
        # This bypasses the wrapper's logic that incorrectly assumes HTTP mode
        real_client = QdrantClient(path=str(storage_dir))
        vector_store.client = real_client
        
        # Step D3: Ensure Collection Exists (Upsert will fail otherwise)
        if not real_client.collection_exists(collection_name):
            logger.info(f"   - Creating collection '{collection_name}' (Dims: 384)...")
            real_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

        # 3. Create Pipeline
        pipeline = IngestionPipeline(
            modules=[parser, splitter, embedder],
            vector_store=vector_store,
            collection_name=collection_name
        )

        # 4. Run Ingestion
        if not data_dir.exists():
            logger.error(f"❌ Data directory '{data_dir}' does not exist.")
            return

        files_to_ingest = [str(f) for f in data_dir.rglob("*") if f.is_file()]
        
        if not files_to_ingest:
            logger.warning("⚠️  No files found to ingest.")
            return

        logger.info(f"🚀 Starting ingestion for {len(files_to_ingest)} files...")
        
        # Run the pipeline
        pipeline.run(file_path=files_to_ingest)
        
        logger.info(f"✅ Ingestion Complete! Index persisted to '{storage_dir}'.")

    except Exception as e:
        logger.critical(f"❌ FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


async def run_ingestion():
    """
    Wrapper function to run the ingestion process.
    """
    build_index()


if __name__ == "__main__":
    build_index()




