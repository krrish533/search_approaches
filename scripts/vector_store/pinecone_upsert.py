import os
import json
import yaml
import argparse
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from scripts.injestion.metadata_extraction import MetadataExtractionPipeline  # Adjust import if needed

class PineconeUpdater:
    def __init__(self, config_path="config/config.yaml", env_path="config/.env"):
        load_dotenv(env_path)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_cfg = self.config.get("pinecone", {})

        self.index_name = pinecone_cfg.get("index_name")

        self.pc = Pinecone(api_key=pinecone_api_key)

        # Directly use the existing index - no creation
        print(f"Using existing Pinecone index: {self.index_name}")
        self.index = self.pc.Index(self.index_name)

        self.metadata_pipeline = MetadataExtractionPipeline()

    def update_document_by_filename(self, filename: str):
        metadata = self.metadata_pipeline.process_single_document(filename)
        if metadata is None:
            print(f"❌ Failed to process document {filename}. Aborting update.")
            return

        doc_id = metadata["filename"]
        if doc_id.endswith(".txt"):
            doc_id = doc_id[:-4]  # Normalize for Pinecone and search
        embedding = metadata["embedding"]

        pinecone_metadata = metadata.copy()
        pinecone_metadata.pop("embedding", None)
        pinecone_metadata.pop("filename", None)

        # Ensure context is present for hybrid/dense search
        if "context" not in pinecone_metadata:
            pinecone_metadata["context"] = ""

        # Convert named_entities to list of strings for Pinecone
        if "named_entities" in pinecone_metadata:
            pinecone_metadata["named_entities"] = [
                ent["text"].strip() if isinstance(ent, dict) and "text" in ent else str(ent)
                for ent in pinecone_metadata["named_entities"]
            ]

        print(f"Deleting old vector for document ID: {doc_id}")
        self.index.delete(ids=[doc_id])

        print(f"Uploading new vector and metadata for document ID: {doc_id}")
        self.index.upsert([(doc_id, embedding, pinecone_metadata)])

        print(f"✅ Document '{doc_id}' updated successfully in Pinecone.")

def main():
    parser = argparse.ArgumentParser(description="Update a document in Pinecone by filename.")
    parser.add_argument("--file", required=True, help="Filename of the updated document (e.g., 'Test.pdf')")
    args = parser.parse_args()

    updater = PineconeUpdater()
    updater.update_document_by_filename(args.file)

if __name__ == "__main__":
    main()
