import os
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

class PineconeUploader:
    def __init__(self):
        load_dotenv("config/.env")

        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_config = self.config.get("pinecone", {})

        self.index_name = pinecone_config.get("index_name")
        self.dimension = pinecone_config.get("dimension")
        self.metric = pinecone_config.get("metric")
        self.cloud = pinecone_config.get("cloud", "aws")
        self.region = pinecone_config.get("region")

        self.pc = Pinecone(api_key=pinecone_api_key)

        existing_indexes = self.pc.list_indexes()
        # if self.index_name not in existing_indexes:
        #     print(f"Creating Pinecone index: {self.index_name}")
        #     self.pc.create_index(
        #         name=self.index_name,
        #         dimension=self.dimension,
        #         metric=self.metric,
        #         spec=ServerlessSpec(
        #             cloud=self.cloud,
        #             region=self.region
        #         )
        #     )
        # else:
        #     print(f"Pinecone index '{self.index_name}' already exists.")

        self.index = self.pc.Index(self.index_name)

        paths = self.config["paths"]
        self.metadata_dir = Path(paths["final_output_dir"])

    def upload_embeddings(self):
        json_files = list(self.metadata_dir.glob("*.json"))
        print(f"Found {len(json_files)} metadata JSON files to upload.")

        batch_size = 50
        for i in range(0, len(json_files), batch_size):
            batch_files = json_files[i:i + batch_size]
            vectors = []

            for jf in batch_files:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                doc_id = data.get("filename")
                vector = data.get("embedding")
                if not doc_id or not vector:
                    continue

                metadata = data.copy()
                metadata.pop("embedding", None)
                metadata.pop("filename", None)

                # Convert nested named_entities list of dicts into list of strings
                if "named_entities" in metadata:
                    named_entities = metadata["named_entities"]
                    metadata["named_entities"] = [ent["text"].strip() for ent in named_entities if "text" in ent]

                vectors.append((doc_id, vector, metadata))

            if vectors:
                print(f"Upserting batch {i // batch_size + 1} with {len(vectors)} vectors...")
                self.index.upsert(vectors)

        print("✅ All embeddings and metadata uploaded successfully.")


if __name__ == "__main__":
    uploader = PineconeUploader()
    uploader.upload_embeddings()
