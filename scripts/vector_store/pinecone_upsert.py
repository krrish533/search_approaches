# import os
# import json
# import yaml
# import argparse
# import time
# from pathlib import Path
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from scripts.injestion.metadata_extraction import MetadataExtractionPipeline  # Adjust import as per your project structure

# class PineconeUpdater:
#     def __init__(self, config_path="config/config.yaml", env_path="config/.env"):
#         load_dotenv(env_path)

#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)

#         pinecone_api_key = os.getenv("PINECONE_API_KEY")
#         pinecone_cfg = self.config.get("pinecone", {})

#         self.index_name = pinecone_cfg.get("index_name")
#         self.dimension = pinecone_cfg.get("dimension")
#         self.metric = pinecone_cfg.get("metric")
#         self.cloud = pinecone_cfg.get("cloud", "aws")
#         self.region = pinecone_cfg.get("region")

#         self.pc = Pinecone(api_key=pinecone_api_key)

#         existing_indexes = self.pc.list_indexes()
#         if self.index_name not in existing_indexes:
#             print(f"Creating Pinecone index: {self.index_name}")
#             self.pc.create_index(
#                 name=self.index_name,
#                 dimension=self.dimension,
#                 metric=self.metric,
#                 spec=ServerlessSpec(
#                     cloud=self.cloud,
#                     region=self.region
#                 )
#             )
#             print("Waiting for index to be ready...")
#             time.sleep(10)  # wait for index readiness
#         else:
#             print(f"Pinecone index '{self.index_name}' already exists. Using existing index.")

#         self.index = self.pc.Index(self.index_name)
#         self.metadata_pipeline = MetadataExtractionPipeline()

#     def update_document_by_filename(self, filename: str):
#         metadata = self.metadata_pipeline.process_single_document(filename)
#         if metadata is None:
#             print(f"❌ Failed to process document {filename}. Aborting update.")
#             return

#         doc_id = metadata["filename"]
#         embedding = metadata["embedding"]

#         pinecone_metadata = metadata.copy()
#         pinecone_metadata.pop("embedding", None)
#         pinecone_metadata.pop("filename", None)

#         # Convert named_entities list of dicts to list of strings for Pinecone metadata
#         if "named_entities" in pinecone_metadata:
#             pinecone_metadata["named_entities"] = [
#                 ent["text"].strip() for ent in pinecone_metadata["named_entities"] if "text" in ent
#             ]

#         print(f"Deleting old vector for document ID: {doc_id}")
#         self.index.delete(ids=[doc_id])

#         print(f"Uploading new vector and metadata for document ID: {doc_id}")
#         self.index.upsert([(doc_id, embedding, pinecone_metadata)])

#         print(f"✅ Document '{doc_id}' updated successfully in Pinecone.")

# def main():
#     parser = argparse.ArgumentParser(description="Update a document in Pinecone by filename.")
#     parser.add_argument("--file", required=True, help="Filename of the updated document (e.g., 'Test.pdf')")
#     args = parser.parse_args()

#     updater = PineconeUpdater()
#     updater.update_document_by_filename(args.file)

# if __name__ == "__main__":
#     main()


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
        embedding = metadata["embedding"]

        pinecone_metadata = metadata.copy()
        pinecone_metadata.pop("embedding", None)
        pinecone_metadata.pop("filename", None)

        if "named_entities" in pinecone_metadata:
            pinecone_metadata["named_entities"] = [
                ent["text"].strip() for ent in pinecone_metadata["named_entities"] if "text" in ent
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
