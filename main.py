from scripts.injestion.metadata_extraction import MetadataExtractionPipeline
from scripts.injestion.embed_to_pinecone import PineconeUploader
from scripts.search.search import DenseRetriever

def main():
    print("Starting metadata extraction...")
    metadata_pipeline = MetadataExtractionPipeline(config_path="config/config.yaml")
    metadata_pipeline.run_pipeline()

    print("\nUploading embeddings to Pinecone...")
    uploader = PineconeUploader()
    uploader.upload_embeddings()

    print("\nReady to search!")
    retriever = DenseRetriever()

    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.strip().lower() == 'exit':
            break
        result = retriever.search(query)
        if result["answer"]:
            print(f"\nAnswer:\n{result['answer']}")
            print(f"Relevance Score: {result['score']:.4f}")
        else:
            print("No relevant documents found.")
        print(f"Query took {result['time']:.2f} seconds")

if __name__ == "__main__":
    main()
