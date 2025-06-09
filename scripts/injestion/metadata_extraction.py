# Metadata Extraction Pipeline for Legal Documents
import os
import json
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from openai import AzureOpenAI
from keybert import KeyBERT

class MetadataExtractionPipeline:
    def __init__(self):
        load_dotenv("config/.env")
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        paths = self.config["paths"]
        metadata_cfg = self.config["metadata"]

        self.pdf_input_dir = Path(paths["pdf_input_dir"])
        self.txt_output_dir = Path(paths["txt_output_dir"])
        self.final_output_dir = Path(paths["final_output_dir"])
        self.intent_example_path = Path(paths["intent_example_path"])

        self.txt_output_dir.mkdir(parents=True, exist_ok=True)
        self.final_output_dir.mkdir(parents=True, exist_ok=True)

        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        self.embedding_model = SentenceTransformer(self.config["embedding_model"]["name"])

        if metadata_cfg["keyword_model"] == "keybert":
            self.keyword_model = KeyBERT()
        else:
            raise ValueError("Only KeyBERT is supported for keyword extraction")

        self.ner_pipeline = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(metadata_cfg["ner_model"]),
            tokenizer=AutoTokenizer.from_pretrained(metadata_cfg["ner_model"]),
            aggregation_strategy="simple"
        )

        self.intent_examples = self.load_intent_examples()

    def load_intent_examples(self):
        with open(self.intent_example_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_text_from_pdf(self, pdf_path):
        reader = PdfReader(str(pdf_path))
        return "\n".join([page.extract_text() or "" for page in reader.pages]).strip()

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^\w\s\-]', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip().lower()

    def extract_keywords(self, text):
        cleaned = self.clean_text(text)
        keywords = [kw for kw, _ in self.keyword_model.extract_keywords(
            cleaned, top_n=10, stop_words='english', keyphrase_ngram_range=(1, 1)
        )]
        return keywords

    def predict_intent(self, text):
        labels = list(self.intent_examples.keys())
        examples = list(self.intent_examples.values())
        doc_emb = self.embedding_model.encode(text, convert_to_tensor=True)
        example_emb = self.embedding_model.encode(examples, convert_to_tensor=True)
        cosine_scores = util.cos_sim(doc_emb, example_emb)[0]
        best_idx = int(cosine_scores.argmax())
        return labels[best_idx], float(cosine_scores[best_idx])

    def extract_entities(self, text):
        return [{"text": ent["word"], "label": ent["entity_group"]} for ent in self.ner_pipeline(text)]

    def summarize(self, text):
        prompt = f"Summarize the following legal document in 2–3 lines:\n\n{text[:3000]}"
        response = self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()

    def embed_text(self, text):
        return self.embedding_model.encode(text).tolist()
    
    def process_single_document(self, filename: str):
        pdf_path = self.pdf_input_dir / filename
        if not pdf_path.exists():
            print(f"❌ File {filename} not found in {self.pdf_input_dir}")
            return None

        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"⚠️ {filename} is empty or unreadable.")
            return None

        cleaned_text = text.strip()
        intent, confidence = self.predict_intent(cleaned_text)
        result = {
            "filename": f"{Path(filename).stem}.txt",
            # "context": cleaned_text,  # Always save full text
            "keywords": self.extract_keywords(cleaned_text),
            "intent_category": intent,
            # "intent_confidence": confidence,  # Optional, still commented out
            "named_entities": self.extract_entities(cleaned_text),
            "summary": self.summarize(cleaned_text),
            "embedding": self.embed_text(cleaned_text)
        }

        # Save JSON metadata for the document
        output_path = self.final_output_dir / f"{Path(filename).stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Saved metadata: {output_path.name}")

        return result

    def run_pipeline(self):
        for pdf_file in self.pdf_input_dir.glob("*.pdf"):
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                print(f"⚠️ {pdf_file.name} is empty or unreadable.")
                continue

            txt_path = self.txt_output_dir / f"{pdf_file.stem}.txt"
            txt_path.write_text(text, encoding="utf-8")
            print(f"📄 Extracted: {pdf_file.name} ➜ {txt_path.name}")

        for txt_file in self.txt_output_dir.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                print(f"⚠️ Skipping {txt_file.name} — empty text.")
                continue

            print(f"🔍 Processing: {txt_file.name}")
            intent, confidence = self.predict_intent(text)
            result = {
                "filename": txt_file.name,
                # "context": text,  # Always save full text
                "keywords": self.extract_keywords(text),
                "intent_category": intent,
                # "intent_confidence": confidence,  # Optional, still commented out
                "named_entities": self.extract_entities(text),
                "summary": self.summarize(text),
                "embedding": self.embed_text(text)
            }

            output_path = self.final_output_dir / f"{txt_file.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"✅ Saved metadata: {output_path.name}")

if __name__ == "__main__":
    pipeline = MetadataExtractionPipeline()
    pipeline.run_pipeline()
