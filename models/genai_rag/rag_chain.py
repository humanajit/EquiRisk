from langchain.prompts import PromptTemplate


class EquiRiskRAGAdvisor:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def build_risk_prompt_template(self) -> PromptTemplate:
        """Constructs a strict prompt instructing the model to focus purely on risk factors."""
        template = """
You are an expert AI Stock Risk Advisor for the EquiRisk platform.
Your goal is to explain investment risks, volatility, and negative news factors to clients.
Do NOT predict or guess target stock prices. Focus strictly on risk quantification.

Context from Recent Financial News & Reports:
{context}

Client Question: {question}

Provide a concise, factual summary highlighting:
1. Primary Risk Drivers (if any)
2. Sentiment & Regulatory/Market Threats
3. Risk Mitigation Advice for the Client

Advisor Response:
"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def answer_risk_query(self, query: str, ticker_symbol: str, top_k: int = 3) -> dict:
        """Retrieves context from FAISS and returns structured context for LLM synthesis."""
        if not self.vector_store:
            return {"error": "Vector store is not initialized."}

        # Similarity search in FAISS vector index
        retrieved_docs = self.vector_store.similarity_search(query, k=top_k)

        context_str = "\n---\n".join([f"[{doc.metadata.get('source')}] {doc.page_content}" for doc in retrieved_docs])
        
        prompt_template = self.build_risk_prompt_template()
        formatted_prompt = prompt_template.format(context=context_str, question=query)

        return {
            "query": query,
            "symbol": ticker_symbol,
            "retrieved_context_chunks": len(retrieved_docs),
            "formatted_prompt": formatted_prompt,
            "docs": [doc.page_content for doc in retrieved_docs]
        }