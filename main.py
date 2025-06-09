from scripts.injestion.metadata_extraction import MetadataExtractionPipeline
from scripts.injestion.embed_to_pinecone import PineconeUploader
from scripts.dense_search.dense_search import DenseRetriever
from scripts.hybrid_search.hybrid_search import HybridSearcher

def main():
    print("Starting metadata extraction...")
    metadata_pipeline = MetadataExtractionPipeline()  # Removed config_path argument
    metadata_pipeline.run_pipeline()

    print("\nUploading embeddings to Pinecone...")
    uploader = PineconeUploader()
    uploader.upload_embeddings()

    print("\nReady to search!")
    mode = None
    while mode not in {"dense", "hybrid"}:
        mode = input("Choose search mode ('dense' or 'hybrid'): ").strip().lower()
    if mode == "dense":
        retriever = DenseRetriever()
    else:
        retriever = HybridSearcher()

    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.strip().lower() == 'exit':
            break
        result = retriever.search(query)
        if result["answer"]:
            print(f"\nAnswer:\n{result['answer']}")
            print(f"Document ID: {result['document_id']}")
            if mode == "dense":
                print(f"Relevance Score: {result['score']:.4f}")
                if result.get("rerank_score") is not None:
                    print(f"Rerank Score (CrossEncoder): {result['rerank_score']:.4f}")
            else:
                print(f"Hybrid Score: {result['hybrid_score']:.4f}")
                print(f"Dense Score: {result['dense_score']:.4f}")
                print(f"Sparse Score: {result['sparse_score']:.4f}")
        else:
            print("No relevant documents found.")
        print(f"Query took {result['time']:.2f} seconds")

if __name__ == "__main__":
    main()
