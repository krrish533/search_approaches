# import os
# import time
# import yaml
# import json
# from dotenv import load_dotenv
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from openai import AzureOpenAI

# class DenseRetriever:
#     def __init__(self, config_path="config/config.yaml"):
#         load_dotenv("config/.env")

#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)

#         # Pinecone setup
#         pinecone_api_key = os.getenv("PINECONE_API_KEY")
#         pinecone_config = self.config.get("pinecone", {})
#         self.index_name = pinecone_config.get("index_name")

#         self.pc = Pinecone(api_key=pinecone_api_key)
#         self.index = self.pc.Index(self.index_name)

#         # Embedding model (for query)
#         embed_model_name = self.config["embedding_model"]["name"]
#         self.embedder = SentenceTransformer(embed_model_name)

#         # Reranker (cross-encoder)
#         reranker_model_name = self.config.get("reranker", {}).get("model")
#         if reranker_model_name:
#             self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
#             self.reranker = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.reranker.to(self.device)
#         else:
#             self.reranker = None

#         # Azure OpenAI client for LLM answer generation
#         self.azure_client = AzureOpenAI(
#             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
#         )
#         self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
  
#         # Metadata directory to load stored JSON metadata with summaries or content
#         self.metadata_dir = self.config["paths"]["final_output_dir"]

#         self.top_k = self.config.get("output", {}).get("return_top_k_docs", 5)

#     def embed_query(self, query):
#         return self.embedder.encode(query).tolist()

#     def rerank(self, query, docs):
#         inputs = self.tokenizer(
#             [f"{query} </s></s> {doc}" for doc in docs],
#             padding=True,
#             truncation=True,
#             return_tensors="pt"
#         ).to(self.device)

#         with torch.no_grad():
#             logits = self.reranker(**inputs).logits
#         #     probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()  # positive class probs

#         # return probs
#             if logits.shape[1] == 1:  # binary classification
#                 scores = logits.squeeze().cpu().tolist()
#             else:  # multi-class classification
#                 probs = torch.softmax(logits, dim=1)
#                 scores = probs[:, 1].cpu().tolist()  # assuming positive class is at index 1
#         return scores
    
#     def generate_answer(self, query, doc_text):
#         prompt = (
#             f"Use the following document text to answer the question below.\n\n"
#             f"Document Text:\n{doc_text}\n\nQuestion:\n{query}\n\nAnswer:"
#         )
#         response = self.azure_client.chat.completions.create(
#             model=self.azure_deployment,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=256
#         )
#         return response.choices[0].message.content.strip()

#     def search(self, query):
#         start_time = time.time()

#         # Step 1: Embed user query
#         query_vec = self.embed_query(query)

#         # Step 2: Query Pinecone index for top_k matches with metadata
#         response = self.index.query(vector=query_vec, top_k=self.top_k, include_metadata=True)

#         matches = response.matches
#         if not matches:
#             return {"answer": None, "document_id": None, "score": 0, "time": time.time() - start_time}

#         # Step 3: Extract summaries or content for reranking
#         docs = [match.metadata.get("summary", "") for match in matches]

#         # Step 4: Apply reranking if reranker exists
#         if self.reranker:
#             rerank_scores = self.rerank(query, docs)
#             best_idx = rerank_scores.index(max(rerank_scores))
#             best_doc_id = matches[best_idx].id
#             best_rerank_score = rerank_scores[best_idx]
#             best_relevance_score = matches[best_idx].score
#         else:
#             best_doc_id = matches[0].id
#             best_score = matches[0].score
#             best_doc_summary = docs[0]

#         # Step 5: Load full document text or summary from metadata JSON file
#         metadata_path = os.path.join(self.metadata_dir, f"{best_doc_id}")
#         if not metadata_path.endswith(".json"):
#             metadata_path += ".json"

#         try:
#             with open(metadata_path, "r", encoding="utf-8") as f:
#                 metadata = json.load(f)
#                 doc_text = metadata.get("summary", "") or metadata.get("context", "")
#         except FileNotFoundError:
#             doc_text = best_doc_summary  # fallback

#         # Step 6: Generate answer with LLM using query and document text
#         answer = self.generate_answer(query, doc_text)

#         total_time = time.time() - start_time

#         return {
#             "answer": answer,
#             "document_id": best_doc_id,
#             "score": best_score,
#             "relevance_score": best_relevance_score,
#             "rerank_score": best_rerank_score,
#             "time": total_time
#         }

# if __name__ == "__main__":
#     retriever = DenseRetriever()
#     user_query = input("Enter your question: ")

#     result = retriever.search(user_query)
#     if result["answer"]:
#         print(f"\nAnswer:\n{result['answer']}\n")
#         print(f"Document ID: {result['document_id']}")
#         print(f"Relevance Score (Pinecone): {result['relevance_score']:.4f}")
#         print(f"Rerank Score (CrossEncoder): {result['rerank_score']:.4f}")
#     else:
#         print("No relevant documents found.")

#     print(f"Time taken: {result['time']:.2f} seconds")


import os
import time
import yaml
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from openai import AzureOpenAI

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
            #print(f"[INFO] Loading reranker: {reranker_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.reranker.to(self.device)
            #print("[INFO] Reranker loaded successfully.")
        else:
            self.reranker = None
            print("[WARNING] No reranker model configured.")

        # Azure OpenAI client for LLM answer generation
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Metadata directory for document JSON files
        self.metadata_dir = self.config["paths"]["final_output_dir"]

        self.top_k = self.config.get("output", {}).get("return_top_k_docs", 5)

    def embed_query(self, query):
        return self.embedder.encode(query).tolist()

    def rerank(self, query, docs):
        #print(f"[DEBUG] Applying reranking to top {len(docs)} docs")
        inputs = self.tokenizer(
            [f"{query} </s></s> {doc}" for doc in docs],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.reranker(**inputs).logits
            if logits.shape[1] == 1:
                scores = torch.sigmoid(logits).squeeze().cpu().tolist()
                # scores = logits.squeeze().cpu().tolist()
            else:
                probs = torch.softmax(logits, dim=1)
                scores = probs[:, 1].cpu().tolist()
        return scores

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
        response = self.index.query(vector=query_vec, top_k=self.top_k, include_metadata=True)
        matches = response.matches

        if not matches:
            return {"answer": None, "document_id": None, "score": 0, "relevance_score": 0, "rerank_score": None, "time": time.time() - start_time}

        docs = [match.metadata.get("summary", "") for match in matches]

        best_doc_id = matches[0].id
        best_relevance_score = matches[0].score
        best_rerank_score = None
        best_doc_summary = docs[0]

        if self.reranker:
            rerank_scores = self.rerank(query, docs)
            best_idx = rerank_scores.index(max(rerank_scores))
            best_doc_id = matches[best_idx].id
            best_rerank_score = rerank_scores[best_idx]
            best_relevance_score = matches[best_idx].score
            best_doc_summary = docs[best_idx]

        metadata_path = os.path.join(self.metadata_dir, f"{best_doc_id}")
        if not metadata_path.endswith(".json"):
            metadata_path += ".json"
        metadata_path = os.path.abspath(metadata_path)

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                doc_text = metadata.get("summary", "") or metadata.get("context", "")
        except FileNotFoundError:
            #print(f"[WARNING] Metadata file not found: {metadata_path}")
            doc_text = best_doc_summary

        answer = self.generate_answer(query, doc_text)
        total_time = time.time() - start_time

        return {
            "answer": answer,
            "document_id": best_doc_id,
            "score": best_rerank_score or best_relevance_score,
            "relevance_score": best_relevance_score,
            "rerank_score": best_rerank_score,
            "time": total_time
        }

if __name__ == "__main__":
    retriever = DenseRetriever()
    user_query = input("Enter your question: ")
    result = retriever.search(user_query)

    if result["answer"]:
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Document ID: {result['document_id']}")
        print(f"Relevance Score (Pinecone): {result['relevance_score']:.4f}")
        print(f"Rerank Score (CrossEncoder): {result['rerank_score']:.4f}" if result["rerank_score"] is not None else "[INFO] Reranking not applied.")
    else:
        print("No relevant documents found.")

    print(f"Time taken: {result['time']:.2f} seconds")
