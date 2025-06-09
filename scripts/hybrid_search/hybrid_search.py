import os
import time
import json
import yaml
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import AzureOpenAI


class HybridSearcher:
    def __init__(self, config_path="config/config.yaml"):
        load_dotenv("config/.env")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.alpha = self.config.get("hybrid", {}).get("alpha", 0.5)
        self.metadata_dir = self.config["paths"]["final_output_dir"]
        model_name = self.config["embedding_model"]["name"]
        self.embedder = SentenceTransformer(model_name)
        rerank_model = self.config["reranker"]["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(rerank_model)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(rerank_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker.to(self.device)
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = self.config["pinecone"]["index_name"]
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index)
        self.vectorizer = TfidfVectorizer()
        self.corpus_texts, self.corpus_ids = self.load_corpus()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_texts)
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.top_k = self.config.get("output", {}).get("return_top_k_docs", 5)

    def load_corpus(self):
        texts = []
        ids = []
        for file in os.listdir(self.metadata_dir):
            if file.endswith(".json"):
                with open(os.path.join(self.metadata_dir, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = data.get("context", "") or data.get("summary", "")
                    doc_id = file.replace(".json", "")
                    if doc_id.endswith(".txt"):
                        doc_id = doc_id[:-4]
                    texts.append(text)
                    ids.append(doc_id)
        return texts, ids

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
            if logits.shape[1] == 1:
                return logits.squeeze().cpu().tolist()
            else:
                probs = torch.softmax(logits, dim=1)
                return probs[:, 1].cpu().tolist()

    def generate_answer(self, query, doc_text):
        prompt = (
            f"Use the following document text to answer the question below.\n\n"
            f"Document Text:\n{doc_text}\n\nQuestion:\n{query}\n\nAnswer:"
        )
        response = self.azure_client.chat.completions.create(
            model=self.azure_deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()

    def search(self, query):
        start_time = time.time()
        query_vec = self.embed_query(query)
        # --- DENSE: Pinecone top_k ---
        dense_response = self.index.query(vector=query_vec, top_k=self.top_k, include_metadata=True)
        dense_matches = dense_response.matches or []
        dense_doc_scores = {m.id: m.score for m in dense_matches}
        # --- SPARSE: TF-IDF top_k ---
        sparse_query_vec = self.vectorizer.transform([query])
        sparse_scores_arr = self.tfidf_matrix.dot(sparse_query_vec.T).toarray().flatten()
        # Get top_k sparse doc indices
        sparse_top_indices = sparse_scores_arr.argsort()[::-1][:self.top_k]
        sparse_doc_scores = {self.corpus_ids[i]: sparse_scores_arr[i] for i in sparse_top_indices if sparse_scores_arr[i] > 0}
        # --- Union of doc_ids ---
        all_doc_ids = set(dense_doc_scores.keys()) | set(sparse_doc_scores.keys())
        doc_scores = []
        for doc_id in all_doc_ids:
            dense_score = dense_doc_scores.get(doc_id, 0.0)
            sparse_score = sparse_doc_scores.get(doc_id, 0.0)
            hybrid_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            doc_scores.append({
                "id": doc_id,
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "hybrid_score": hybrid_score
            })
        if not doc_scores:
            print("[WARNING] No results found from either dense or sparse search.")
            return {"answer": None, "document_id": None, "score": 0, "time": time.time() - start_time}
        # Sort by hybrid score
        top_doc = sorted(doc_scores, key=lambda x: x["hybrid_score"], reverse=True)[0]
        best_doc_id = top_doc["id"]
        # --- FIX: Strip .txt from doc_id before appending .json ---
        base_id = best_doc_id
        if base_id.endswith(".txt"):
            base_id = base_id[:-4]
        metadata_path = os.path.join(self.metadata_dir, base_id + ".json")
        doc_text = ""
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                doc_text = metadata.get("context", "") or metadata.get("summary", "") or ""
                print(f"[DEBUG] Loaded document from metadata: {base_id}")
        except FileNotFoundError:
            print(f"[WARNING] Metadata file not found: {metadata_path}")
        if not doc_text:
            try:
                idx = self.corpus_ids.index(best_doc_id)
                doc_text = self.corpus_texts[idx]
                print(f"[INFO] Fallback to TF-IDF text for document: {best_doc_id}")
            except ValueError:
                print(f"[ERROR] No text found for document: {best_doc_id}")
                doc_text = ""
        print(f"[DEBUG] Loaded doc text length: {len(doc_text)}")
        answer = self.generate_answer(query, doc_text)
        total_time = time.time() - start_time
        return {
            "answer": answer,
            "document_id": best_doc_id,
            "hybrid_score": top_doc["hybrid_score"],
            "dense_score": top_doc["dense_score"],
            "sparse_score": top_doc["sparse_score"],
            "time": total_time
        }


if __name__ == "__main__":
    hs = HybridSearcher()
    query = input("Enter your question: ")
    result = hs.search(query)
    if result["answer"]:
        print("\nAnswer:\n", result["answer"])
        print("\nTop Document ID:", result["document_id"])
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"Dense Score: {result['dense_score']:.4f}")
        print(f"Sparse Score: {result['sparse_score']:.4f}")
    else:
        print("No relevant documents found.")
    print(f"Time taken: {result['time']:.2f} seconds")
