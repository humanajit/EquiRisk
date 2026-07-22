import os
import io
import boto3
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class RiskVectorStoreManager:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initializes HuggingFace Embeddings for vector search."""
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def create_vector_store_from_df(self, df: pd.DataFrame, text_column: str = "full_text") -> FAISS:
        """Converts DataFrame records into LangChain Documents and builds a FAISS index."""
        if df.empty or text_column not in df.columns:
            print("Error: Empty DataFrame or missing text column.")
            return None

        documents = []
        for _, row in df.iterrows():
            text = row[text_column]
            if isinstance(text, str) and text.strip():
                # Store extra metadata for rich context during RAG retrieval
                metadata = {
                    "symbol": row.get("symbol", "UNKNOWN"),
                    "source": row.get("source_name", "GNews"),
                    "published_at": str(row.get("publishedAt", "")),
                    "sentiment_label": row.get("risk_sentiment_label", "NEUTRAL")
                }
                documents.append(Document(page_content=text, metadata=metadata))

        print(f"Indexing {len(documents)} document chunks into FAISS...")
        vector_store = FAISS.from_documents(documents, self.embeddings)
        return vector_store

    def save_index_to_s3(self, vector_store: FAISS, s3_prefix: str):
        """Saves FAISS index locally and uploads the index files to S3."""
        local_dir = "/tmp/faiss_index"
        os.makedirs(local_dir, exist_ok=True)
        vector_store.save_local(local_dir)

        s3_client = boto3.client("s3")
        bucket_name = os.getenv("S3_BUCKET_NAME", "equirisk-data-lake")

        for file_name in os.listdir(local_dir):
            local_path = os.path.join(local_dir, file_name)
            s3_key = f"{s3_prefix}/{file_name}"
            s3_client.upload_file(local_path, bucket_name, s3_key)
            print(f"Uploaded index asset: s3://{bucket_name}/{s3_key}")


if __name__ == "__main__":
    # Test creation with sample news snippets
    sample_news = pd.DataFrame([
        {
            "full_text": "Reliance faces short-term volatility due to global crude oil price fluctuations.",
            "symbol": "RELIANCE",
            "source_name": "GNews",
            "risk_sentiment_label": "HIGH_RISK_SENTIMENT"
        }
    ])
    
    manager = RiskVectorStoreManager()
    vs = manager.create_vector_store_from_df(sample_news)
    print("FAISS Index initialization test successful!")