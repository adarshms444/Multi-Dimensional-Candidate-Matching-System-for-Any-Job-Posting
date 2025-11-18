
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import config
from utils import simple_tokenizer
import chromadb
from chromadb.utils import embedding_functions

class HybridRetriever:
    def __init__(self):
        self.bm25_index = None
        self.corpus_ids = []
        
        self.sbert_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        self.chroma_client = chromadb.Client() 
        
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="resume_collection",
            embedding_function=self.embedding_func
        )
        print("HybridRetriever initialized with ChromaDB.")

    def index(self, corpus: list[str], corpus_ids: list[str]):
        """Creates the BM25 and ChromaDB indexes."""
        if not corpus:
            print("Warning: No documents to index.")
            return

        print(f"Indexing {len(corpus)} documents...")
        self.corpus_ids = corpus_ids
        
        tokenized_corpus = [simple_tokenizer(doc) for doc in corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        
        existing_ids = self.collection.get()['ids']
        if existing_ids:
            print(f"Clearing {len(existing_ids)} old items from ChromaDB.")
            self.collection.delete(ids=existing_ids)
        
        self.collection.add(
            documents=corpus,
            ids=corpus_ids
        )
        print("Indexing complete.")

    def search(self, query: str, top_k: int) -> list[str]:
        """
        Performs hybrid search.
        1. Gets top_k from Sparse (BM25).
        2. Gets top_k from Dense (Chroma).
        3. Returns the UNION of the two lists.
        """
        if self.bm25_index is None:
            raise Exception("Must call .index() before .search()")
            
        print(f"Running hybrid search for query: {query[:50]}...")
        
        tokenized_query = simple_tokenizer(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
  
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_ids = [self.corpus_ids[i] for i in bm25_top_indices]
        print(f"BM25 found IDs: {sparse_ids}")

        dense_results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        dense_ids = dense_results['ids'][0]
        print(f"ChromaDB found IDs: {dense_ids}")

        fused_ids = list(set(sparse_ids) | set(dense_ids))
        
        print(f"Retrieval found {len(fused_ids)} unique candidates for re-ranking.")
        return fused_ids

print("File 'retrieval.py' (Upgraded with ChromaDB Fix) created.")
