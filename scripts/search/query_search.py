import os
import time
import yaml
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DenseRetriever:
    def __init__(self, config_path="config/config.yaml"):
        load_dotenv("config/.env")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Pinecone setup
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_config = self.config.get("pinecone", {})
        self.index_name = pinecone_config.get("index_name")

        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(self.index_name)

        # Embedding model (for query)
        embed_model_name = self.config["embedding_model"]["name"]
        self.embedder = SentenceTransformer(embed_model_name)

        # Reranker (cross-encoder)
        reranker_model_name = self.config.get("reranker", {}).get("model")
        if reranker_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.reranker.to(self.device)
        else:
            self.reranker = None

        self.top_k = self.config.get("output", {}).get("return_top_k_docs", 5)

    def embed_query(self, query):
        return self.embedder.encode(query).tolist()

    def rerank(self, query, docs):
        inputs = self.tokenizer(
            [f"{query} </s></s> {doc}" for doc in docs],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.reranker(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()  # positive class probs

        return probs

    def search(self, query):
        start_time = time.time()

        # Step 1: Embed user query
        query_vec = self.embed_query(query)

        # Step 2: Query Pinecone index for top_k matches
        response = self.index.query(vector=query_vec, top_k=self.top_k, include_metadata=True)

        matches = response.matches
        if not matches:
            return {"answer": None, "score": 0, "time": time.time() - start_time}

        # Step 3: Extract summaries or content for reranking
        docs = [match.metadata.get("summary", "") for match in matches]

        # Step 4: Apply reranking if reranker model exists
        if self.reranker:
            rerank_scores = self.rerank(query, docs)
            # Find best scoring document
            best_idx = rerank_scores.index(max(rerank_scores))
            answer = docs[best_idx]
            score = rerank_scores[best_idx]
        else:
            # If no reranker, pick top Pinecone match
            answer = docs[0]
            score = matches[0].score

        total_time = time.time() - start_time

        return {
            "answer": answer,
            "score": score,
            "time": total_time
        }

if __name__ == "__main__":
    retriever = DenseRetriever()
    user_query = input("Enter your question: ")

    result = retriever.search(user_query)
    if result["answer"]:
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Relevance Score: {result['score']:.4f}")
    else:
        print("No relevant documents found.")

    print(f"Time taken: {result['time']:.2f} seconds")
