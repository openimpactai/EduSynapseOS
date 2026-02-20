# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Knowledge RAG for System Explainer.

This module provides semantic search over the EduSynapseOS knowledge base
documentation. Instead of loading all documents into context, it retrieves
only the most relevant chunks based on the user's query.

Architecture:
    1. Documents are chunked by sections (~500-1000 chars)
    2. Each chunk is embedded and stored in Qdrant
    3. User queries retrieve top-k most similar chunks
    4. Only relevant chunks are sent to LLM

Example:
    rag = KnowledgeRAG(qdrant_client, embedding_service)
    await rag.index_knowledge_base()  # One-time indexing
    chunks = await rag.retrieve("What are the benefits of EduSynapse?")
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.intelligence.embeddings import EmbeddingService
from src.infrastructure.vectors import QdrantVectorClient, SearchResult

logger = logging.getLogger(__name__)

# Collection name for system knowledge
SYSTEM_KNOWLEDGE_COLLECTION = "system_knowledge"

# Chunk settings
MIN_CHUNK_SIZE = 200  # Minimum characters per chunk
MAX_CHUNK_SIZE = 1500  # Maximum characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks for context continuity


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge base content.

    Attributes:
        id: Unique identifier for the chunk.
        content: The text content of the chunk.
        document: Source document name.
        section: Section title within the document.
        metadata: Additional metadata.
    """
    id: str
    content: str
    document: str
    section: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    """Retrieved chunk with similarity score.

    Attributes:
        chunk: The knowledge chunk.
        score: Similarity score (0-1).
    """
    chunk: KnowledgeChunk
    score: float


class KnowledgeRAG:
    """RAG retriever for System Explainer knowledge base.

    This class handles:
    - Indexing knowledge base documents into Qdrant
    - Retrieving relevant chunks based on user queries
    - Formatting chunks for LLM context

    Attributes:
        _qdrant: Qdrant vector client.
        _embedding: Embedding service for text vectorization.
    """

    def __init__(
        self,
        qdrant_client: QdrantVectorClient,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialize Knowledge RAG.

        Args:
            qdrant_client: Qdrant client for vector operations.
            embedding_service: Service for generating embeddings.
        """
        self._qdrant = qdrant_client
        self._embedding = embedding_service
        self._kb_dir = Path("docs/KNOWLEDGE-BASE")

    def _generate_chunk_id(self, document: str, section: str, content: str) -> str:
        """Generate deterministic chunk ID.

        Args:
            document: Document name.
            section: Section title.
            content: Chunk content.

        Returns:
            MD5 hash of the combined content.
        """
        combined = f"{document}:{section}:{content[:100]}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _chunk_document(self, content: str, document: str) -> list[KnowledgeChunk]:
        """Split document into chunks by sections.

        Strategy:
        1. Split by ## headers (sections)
        2. If section is too large, split by paragraphs
        3. Ensure each chunk has context (header info)

        Args:
            content: Full document content.
            document: Document name.

        Returns:
            List of KnowledgeChunk objects.
        """
        chunks: list[KnowledgeChunk] = []

        # Extract document title (first # header)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else document

        # Split by ## headers
        sections = re.split(r'\n(?=##\s)', content)

        for section in sections:
            if not section.strip():
                continue

            # Extract section title
            section_match = re.match(r'^##\s*(.+)$', section, re.MULTILINE)
            section_title = section_match.group(1) if section_match else "Overview"

            # Clean the section content
            section_content = section.strip()

            # If section is small enough, keep as one chunk
            if len(section_content) <= MAX_CHUNK_SIZE:
                if len(section_content) >= MIN_CHUNK_SIZE:
                    chunk_id = self._generate_chunk_id(document, section_title, section_content)
                    chunks.append(KnowledgeChunk(
                        id=chunk_id,
                        content=section_content,
                        document=document,
                        section=section_title,
                        metadata={
                            "doc_title": doc_title,
                            "char_count": len(section_content),
                        }
                    ))
            else:
                # Split large sections by paragraphs or subsections
                sub_chunks = self._split_large_section(
                    section_content,
                    document,
                    section_title,
                    doc_title
                )
                chunks.extend(sub_chunks)

        return chunks

    def _split_large_section(
        self,
        content: str,
        document: str,
        section_title: str,
        doc_title: str,
    ) -> list[KnowledgeChunk]:
        """Split a large section into smaller chunks.

        Args:
            content: Section content.
            document: Document name.
            section_title: Section title.
            doc_title: Document title.

        Returns:
            List of chunks from the section.
        """
        chunks: list[KnowledgeChunk] = []

        # Try splitting by ### subsections first
        subsections = re.split(r'\n(?=###\s)', content)

        if len(subsections) > 1:
            # Process each subsection
            for subsection in subsections:
                if len(subsection.strip()) >= MIN_CHUNK_SIZE:
                    sub_match = re.match(r'^###\s*(.+)$', subsection, re.MULTILINE)
                    sub_title = f"{section_title} > {sub_match.group(1)}" if sub_match else section_title

                    chunk_id = self._generate_chunk_id(document, sub_title, subsection)
                    chunks.append(KnowledgeChunk(
                        id=chunk_id,
                        content=subsection.strip(),
                        document=document,
                        section=sub_title,
                        metadata={
                            "doc_title": doc_title,
                            "char_count": len(subsection),
                        }
                    ))
        else:
            # Split by double newlines (paragraphs)
            paragraphs = content.split('\n\n')
            current_chunk = ""
            chunk_num = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Check if adding this paragraph exceeds max size
                if len(current_chunk) + len(para) + 2 > MAX_CHUNK_SIZE:
                    # Save current chunk if it's large enough
                    if len(current_chunk) >= MIN_CHUNK_SIZE:
                        chunk_num += 1
                        chunk_id = self._generate_chunk_id(document, f"{section_title}_{chunk_num}", current_chunk)
                        chunks.append(KnowledgeChunk(
                            id=chunk_id,
                            content=current_chunk,
                            document=document,
                            section=f"{section_title} (part {chunk_num})",
                            metadata={
                                "doc_title": doc_title,
                                "char_count": len(current_chunk),
                                "part": chunk_num,
                            }
                        ))
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

            # Don't forget the last chunk
            if len(current_chunk) >= MIN_CHUNK_SIZE:
                chunk_num += 1
                chunk_id = self._generate_chunk_id(document, f"{section_title}_{chunk_num}", current_chunk)
                chunks.append(KnowledgeChunk(
                    id=chunk_id,
                    content=current_chunk,
                    document=document,
                    section=f"{section_title} (part {chunk_num})" if chunk_num > 1 else section_title,
                    metadata={
                        "doc_title": doc_title,
                        "char_count": len(current_chunk),
                    }
                ))

        return chunks

    async def index_knowledge_base(self, force_reindex: bool = False) -> int:
        """Index all knowledge base documents into Qdrant.

        Args:
            force_reindex: If True, delete existing collection and reindex.

        Returns:
            Number of chunks indexed.
        """
        # Check if collection exists
        exists = await self._qdrant.collection_exists(SYSTEM_KNOWLEDGE_COLLECTION)

        if exists and not force_reindex:
            logger.info("Knowledge base collection already exists. Skipping indexing.")
            return 0

        if exists and force_reindex:
            logger.info("Force reindex requested. Deleting existing collection.")
            await self._qdrant.delete_collection(SYSTEM_KNOWLEDGE_COLLECTION)

        # Create collection
        await self._qdrant.create_collection(
            collection_name=SYSTEM_KNOWLEDGE_COLLECTION,
            vector_size=self._embedding.dimension,
            distance="Cosine",
        )

        # Document order
        doc_order = [
            "INDEX.md",
            "01-EXECUTIVE-SUMMARY.md",
            "02-MEMORY-SYSTEM.md",
            "03-EDUCATIONAL-THEORIES.md",
            "04-AI-AGENTS.md",
            "05-WORKFLOWS.md",
            "06-CURRICULUM-STRUCTURE.md",
            "07-DIAGNOSTICS-ANALYTICS.md",
            "08-EMOTIONAL-INTELLIGENCE.md",
            "09-MULTI-TENANT-ARCHITECTURE.md",
            "10-GAMING-MODULE.md",
            "11-TECHNICAL-STACK.md",
            "12-BUSINESS-MODEL.md",
            "13-SECURITY-COMPLIANCE.md",
            "14-COMPETITIVE-ANALYSIS.md",
        ]

        all_chunks: list[KnowledgeChunk] = []

        # Process each document
        for doc_name in doc_order:
            doc_path = self._kb_dir / doc_name
            if not doc_path.exists():
                logger.warning("Document not found: %s", doc_path)
                continue

            content = doc_path.read_text(encoding="utf-8")
            chunks = self._chunk_document(content, doc_name)
            all_chunks.extend(chunks)
            logger.info("Chunked %s into %d chunks", doc_name, len(chunks))

        if not all_chunks:
            logger.warning("No chunks to index")
            return 0

        # Generate embeddings in batches
        logger.info("Generating embeddings for %d chunks...", len(all_chunks))
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = await self._embedding.embed_batch(chunk_texts, show_progress=True)

        # Prepare points for Qdrant
        points = []
        for chunk, embedding in zip(all_chunks, embeddings):
            points.append({
                "id": chunk.id,
                "vector": embedding,
                "payload": {
                    "content": chunk.content,
                    "document": chunk.document,
                    "section": chunk.section,
                    **chunk.metadata,
                }
            })

        # Upsert to Qdrant
        await self._qdrant.upsert(
            collection_name=SYSTEM_KNOWLEDGE_COLLECTION,
            points=points,
        )

        logger.info("Successfully indexed %d chunks into Qdrant", len(points))
        return len(points)

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant knowledge chunks for a query.

        Args:
            query: User's question or search query.
            limit: Maximum number of chunks to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of retrieved chunks with scores.
        """
        # Check if collection exists
        exists = await self._qdrant.collection_exists(SYSTEM_KNOWLEDGE_COLLECTION)
        if not exists:
            logger.warning("Knowledge base not indexed. Call index_knowledge_base() first.")
            return []

        # Generate query embedding
        try:
            query_embedding = await self._embedding.embed_text(query)
        except Exception as e:
            logger.error("Failed to embed query: %s", e)
            return []

        # Search in Qdrant
        try:
            results: list[SearchResult] = await self._qdrant.search(
                collection_name=SYSTEM_KNOWLEDGE_COLLECTION,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score,
            )
        except Exception as e:
            logger.error("Failed to search knowledge base: %s", e)
            return []

        # Convert to RetrievedChunk
        chunks: list[RetrievedChunk] = []
        for result in results:
            if not result.payload:
                continue

            chunk = KnowledgeChunk(
                id=result.id,
                content=result.payload.get("content", ""),
                document=result.payload.get("document", ""),
                section=result.payload.get("section", ""),
                metadata={
                    k: v for k, v in result.payload.items()
                    if k not in ("content", "document", "section")
                }
            )
            chunks.append(RetrievedChunk(chunk=chunk, score=result.score))

        logger.debug(
            "Retrieved %d chunks for query '%s...' (min_score=%.2f)",
            len(chunks),
            query[:50],
            min_score,
        )

        return chunks

    def format_context(
        self,
        chunks: list[RetrievedChunk],
        max_length: int = 8000,
    ) -> str:
        """Format retrieved chunks for LLM context.

        Args:
            chunks: Retrieved chunks with scores.
            max_length: Maximum total context length.

        Returns:
            Formatted context string.
        """
        if not chunks:
            return ""

        parts = []
        current_length = 0

        for retrieved in chunks:
            chunk = retrieved.chunk

            # Format chunk with source info
            chunk_text = f"[Source: {chunk.document} - {chunk.section}]\n{chunk.content}"

            # Check length
            if current_length + len(chunk_text) > max_length:
                break

            parts.append(chunk_text)
            current_length += len(chunk_text) + 2  # +2 for separator

        return "\n\n---\n\n".join(parts)

    async def is_indexed(self) -> bool:
        """Check if knowledge base is already indexed.

        Returns:
            True if collection exists and has documents.
        """
        try:
            exists = await self._qdrant.collection_exists(SYSTEM_KNOWLEDGE_COLLECTION)
            return exists
        except Exception:
            return False
