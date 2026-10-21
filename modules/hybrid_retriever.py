import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

DEFAULT_DATA_PATH = "data"
DEFAULT_PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "langchain-ned-data")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def document_key(doc: Document) -> str:
    metadata = doc.metadata or {}

    if metadata.get("chunk_id"):
        return str(metadata["chunk_id"])

    source = metadata.get("source", "unknown")
    page = metadata.get("page", "unknown")
    digest = hashlib.sha1(doc.page_content[:500].encode("utf-8")).hexdigest()[:12]
    return f"{source}:{page}:{digest}"


def clone_document(doc: Document, metadata_updates: Optional[Dict] = None) -> Document:
    metadata = dict(doc.metadata or {})
    if metadata_updates:
        metadata.update(metadata_updates)
    return Document(page_content=doc.page_content, metadata=metadata)


def add_chunk_metadata(documents: Sequence[Document]) -> List[Document]:
    source_counts = defaultdict(int)
    updated_documents = []

    for doc in documents:
        metadata = dict(doc.metadata or {})
        source = os.path.basename(str(metadata.get("source", "unknown")))
        page = metadata.get("page")
        page_label = page + 1 if isinstance(page, int) else "unknown"
        source_key = f"{source}_p{page_label}"
        chunk_number = source_counts[source_key]
        source_counts[source_key] += 1

        metadata.update(
            {
                "source": source,
                "page": page_label,
                "chunk_id": f"{source_key}_c{chunk_number}",
                "filename": source,
            }
        )
        updated_documents.append(Document(page_content=doc.page_content, metadata=metadata))

    return updated_documents


def load_pdf_chunks(
    data_path: str = DEFAULT_DATA_PATH,
    chunk_size: int = 300,
    chunk_overlap: int = 100,
) -> List[Document]:
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return add_chunk_metadata(text_splitter.split_documents(documents))


@dataclass
class BM25SearchResult:
    document: Document
    score: float
    rank: int


class BM25Retriever:
    def __init__(self, documents: Sequence[Document]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "Install rank-bm25 before using BM25 retrieval: pip install rank-bm25"
            ) from exc

        self.documents = list(documents)
        self.tokenized_corpus = [tokenize(doc.page_content) for doc in self.documents]
        self.index = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_pdf_dir(
        cls,
        data_path: str = DEFAULT_DATA_PATH,
        chunk_size: int = 300,
        chunk_overlap: int = 100,
    ) -> "BM25Retriever":
        return cls(load_pdf_chunks(data_path, chunk_size, chunk_overlap))

    def search(self, query: str, top_k: int = 8) -> List[BM25SearchResult]:
        query_tokens = tokenize(query)
        scores = self.index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

        results = []
        for rank, (doc_index, score) in enumerate(ranked[:top_k], start=1):
            document = clone_document(
                self.documents[doc_index],
                {"bm25_score": float(score), "bm25_rank": rank},
            )
            results.append(BM25SearchResult(document=document, score=float(score), rank=rank))

        return results


def get_openai_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
    kwargs = {"model": model}
    if openai_key:
        kwargs["api_key"] = openai_key
    return OpenAIEmbeddings(**kwargs)


def get_pinecone_vectorstore(
    index_name: str = DEFAULT_PINECONE_INDEX,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=index_name,
        embedding=get_openai_embeddings(embedding_model),
    )


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Tuple[str, Sequence[Document]]],
    rrf_k: int = 60,
    top_k: int = 10,
) -> List[Document]:
    scores = defaultdict(float)
    documents_by_key = {}
    source_ranks = defaultdict(dict)

    for source_name, documents in ranked_lists:
        for rank, doc in enumerate(documents, start=1):
            key = document_key(doc)
            scores[key] += 1.0 / (rrf_k + rank)
            documents_by_key.setdefault(key, doc)
            source_ranks[key][source_name] = rank

    fused_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        clone_document(
            documents_by_key[key],
            {
                "rrf_score": scores[key],
                "retrieval_ranks": dict(source_ranks[key]),
            },
        )
        for key in fused_keys
    ]


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers before using reranking: "
                "pip install sentence-transformers"
            ) from exc

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 4) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        scored_documents = [
            clone_document(doc, {"reranker_score": float(score)})
            for doc, score in zip(documents, scores)
        ]
        return sorted(
            scored_documents,
            key=lambda doc: doc.metadata.get("reranker_score", 0.0),
            reverse=True,
        )[:top_k]


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_vectorstore: Optional[PineconeVectorStore] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        bm25_k: int = 8,
        dense_k: int = 8,
        fusion_k: int = 10,
        final_k: int = 4,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_vectorstore = dense_vectorstore
        self.reranker = reranker
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.fusion_k = fusion_k
        self.final_k = final_k

    @classmethod
    def from_pdf_dir(
        cls,
        data_path: str = DEFAULT_DATA_PATH,
        use_pinecone: bool = True,
        use_reranker: bool = False,
    ) -> "HybridRetriever":
        bm25_retriever = BM25Retriever.from_pdf_dir(data_path)
        dense_vectorstore = get_pinecone_vectorstore() if use_pinecone else None
        reranker = CrossEncoderReranker() if use_reranker else None
        return cls(
            bm25_retriever=bm25_retriever,
            dense_vectorstore=dense_vectorstore,
            reranker=reranker,
        )

    def retrieve(self, query: str) -> List[Document]:
        bm25_documents = [
            result.document
            for result in self.bm25_retriever.search(query, top_k=self.bm25_k)
        ]

        ranked_lists: List[Tuple[str, Iterable[Document]]] = [("bm25", bm25_documents)]

        if self.dense_vectorstore is not None:
            dense_documents = self.dense_vectorstore.similarity_search(query, k=self.dense_k)
            dense_documents = [
                clone_document(doc, {"dense_rank": rank})
                for rank, doc in enumerate(dense_documents, start=1)
            ]
            ranked_lists.append(("dense", dense_documents))

        fused_documents = reciprocal_rank_fusion(ranked_lists, top_k=self.fusion_k)

        if self.reranker is not None:
            return self.reranker.rerank(query, fused_documents, top_k=self.final_k)

        return fused_documents[: self.final_k]

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retrieve(query)
