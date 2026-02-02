import json
import logging
import time
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from src.pipeline import RAGService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "test_data" / "golden_dataset.json"

def calculate_similarity(text1, text2):
    """
    Calculates cosine similarity between two strings using TF-IDF.
    """
    if not text1 or not text2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        # Handles cases where text is empty or stop words only
        return 0.0

def evaluate_rag():
    """
    Runs the RAG evaluation loop with Rate Limit Handling.
    """
    if not GOLDEN_DATASET_PATH.exists():
        logger.error(f"Dataset not found at {GOLDEN_DATASET_PATH}")
        return

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info("🚀 Starting RAG Evaluation...")
    
    try:
        rag = RAGService()
    except Exception as e:
        logger.error(f"Failed to initialize RAGService: {e}")
        return

    total_hit_rate = 0
    total_similarity = 0
    count = 0

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]
        keywords = item["expected_context_keywords"]

        logger.info(f"[{i+1}/{len(dataset)}] Evaluating Query: {question}")

        # --- RETRY LOOP FOR RATE LIMITS ---
        while True:
            try:
                # 1. Run Pipeline requesting sources
                response_text, retrieved_chunks = rag.query(question, return_sources=True)

                # 2. Check for Quota Error (Based on your pipeline's error string)
                if "Quota exceeded" in response_text or "429" in response_text:
                    logger.warning("⏳ Rate Limit Hit (429). Sleeping for 60 seconds...")
                    time.sleep(60) # Wait for quota to reset (Free tier resets every minute usually)
                    continue # Retry the SAME question
                
                # If we get here, the response is valid (or a different error)
                break 

            except Exception as e:
                logger.error(f"Critical Error in pipeline: {e}")
                response_text = ""
                retrieved_chunks = []
                break
        # ----------------------------------

        # 3. Extract text from chunks (Safe Mode)
        full_retrieved_context = ""
        for chunk in retrieved_chunks:
            if hasattr(chunk, "text"):
                full_retrieved_context += f" {chunk.text}"
            elif hasattr(chunk, "payload") and isinstance(chunk.payload, dict):
                full_retrieved_context += f" {chunk.payload.get('text', '')}"
            else:
                full_retrieved_context += f" {str(chunk)}"

        # 4. Hit Rate Calculation
        hits = sum(1 for k in keywords if k.lower() in full_retrieved_context.lower())
        hit_rate = hits / len(keywords) if keywords else 0
        total_hit_rate += hit_rate

        # 5. Response Similarity
        similarity = calculate_similarity(response_text, ground_truth)
        total_similarity += similarity

        logger.info(f"   -> Hit Rate: {hit_rate:.2f}")
        logger.info(f"   -> Similarity: {similarity:.2f}")
        count += 1
        
        # Optional: Small sleep between successful requests to be polite
        time.sleep(2) 

    if count == 0:
        return

    avg_hit_rate = total_hit_rate / count
    avg_similarity = total_similarity / count

    logger.info("------------------------------------------------")
    logger.info(f"📊 Evaluation Results ({count} queries):")
    logger.info(f"   Average Hit Rate: {avg_hit_rate:.2f}")
    logger.info(f"   Average Response Similarity: {avg_similarity:.2f}")
    logger.info("------------------------------------------------")

if __name__ == "__main__":
    evaluate_rag()