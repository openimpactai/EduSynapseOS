# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Result reranker for RAG context optimization.

This module implements the ResultReranker which uses an LLM to rerank
retrieval results based on their relevance to the query. This improves
context quality beyond pure embedding similarity.

The reranker can operate in two modes:
- LLM-based: Uses the LLM to score relevance (slower, more accurate)
- Heuristic: Uses metadata-based scoring (faster, good for fallback)

Example:
    reranker = ResultReranker(llm_client=llm_client)

    # Rerank results
    reranked = await reranker.rerank(
        query="How do fractions work?",
        results=retrieval_results,
        top_k=5,
    )
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.core.intelligence.llm.client import LLMClient
from src.core.memory.rag.retriever import RetrievalResult, RetrievalSource

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result after reranking.

    Attributes:
        original: The original retrieval result.
        rerank_score: New relevance score from reranking.
        original_rank: Original position in results.
        new_rank: Position after reranking.
    """

    original: RetrievalResult
    rerank_score: float
    original_rank: int
    new_rank: int


class RerankerError(Exception):
    """Exception raised for reranker operations.

    Attributes:
        message: Error description.
        original_error: Original exception if any.
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class ResultReranker:
    """LLM-based result reranker for improved relevance.

    Uses the LLM to score retrieval results by relevance to the query,
    with fallback to heuristic scoring when LLM is unavailable.

    Attributes:
        llm_client: Client for LLM API calls.
        use_llm: Whether to use LLM-based reranking.

    Example:
        reranker = ResultReranker(llm_client=llm)

        reranked = await reranker.rerank(
            query="explain multiplication",
            results=retrieval_results,
            top_k=5,
        )
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        use_llm: bool = True,
    ) -> None:
        """Initialize the reranker.

        Args:
            llm_client: Client for LLM API calls. Required for LLM-based reranking.
            use_llm: Whether to use LLM-based reranking (vs heuristic).
        """
        self._llm = llm_client
        self._use_llm = use_llm and llm_client is not None

        if use_llm and llm_client is None:
            logger.warning(
                "LLM client not provided, falling back to heuristic reranking"
            )

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
        context: str | None = None,
    ) -> list[RerankResult]:
        """Rerank retrieval results by relevance.

        Args:
            query: The original search query.
            results: List of retrieval results to rerank.
            top_k: Number of top results to return.
            context: Optional additional context for ranking.

        Returns:
            List of RerankResult ordered by relevance.
        """
        if not results:
            return []

        if self._use_llm and self._llm:
            try:
                return await self._rerank_with_llm(
                    query=query,
                    results=results,
                    top_k=top_k,
                    context=context,
                )
            except Exception as e:
                logger.warning(
                    "LLM reranking failed, falling back to heuristic: %s", str(e)
                )

        return self._rerank_heuristic(
            query=query,
            results=results,
            top_k=top_k,
        )

    async def _rerank_with_llm(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
        context: str | None,
    ) -> list[RerankResult]:
        """Rerank using LLM for relevance scoring.

        Args:
            query: The original search query.
            results: List of retrieval results to rerank.
            top_k: Number of top results to return.
            context: Optional additional context.

        Returns:
            List of RerankResult with LLM-based scores.
        """
        # Build prompt for batch scoring
        passages = []
        for i, result in enumerate(results):
            source_label = result.source.value.capitalize()
            passages.append(f"[{i+1}] ({source_label}) {result.content[:200]}")

        passages_text = "\n".join(passages)

        prompt = f"""Rate the relevance of each passage to the query on a scale of 0-10.
Consider:
- How directly the passage addresses the query
- Whether the information is accurate and useful
- For student memories, how they might help personalize the response

Query: {query}
{f"Context: {context}" if context else ""}

Passages:
{passages_text}

Return only a JSON array of scores in order, e.g., [8, 5, 9, ...]
Do not include any other text."""

        try:
            response = await self._llm.complete(prompt)
            content = response.content.strip()

            # Parse JSON array of scores
            import json

            # Handle potential markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            scores = json.loads(content)

            if not isinstance(scores, list) or len(scores) != len(results):
                raise ValueError(f"Expected {len(results)} scores, got {len(scores)}")

            # Normalize scores to 0-1
            max_score = max(scores) if max(scores) > 0 else 1
            normalized_scores = [s / max_score for s in scores]

        except Exception as e:
            logger.warning("Failed to parse LLM scores: %s", str(e))
            # Fall back to original scores
            normalized_scores = [r.score for r in results]

        # Build rerank results
        scored = [
            (i, results[i], normalized_scores[i]) for i in range(len(results))
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[2], reverse=True)

        reranked = [
            RerankResult(
                original=result,
                rerank_score=score,
                original_rank=original_rank,
                new_rank=new_rank,
            )
            for new_rank, (original_rank, result, score) in enumerate(scored[:top_k])
        ]

        logger.debug(
            "Reranked %d results with LLM, returning top %d",
            len(results),
            len(reranked),
        )

        return reranked

    def _rerank_heuristic(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RerankResult]:
        """Rerank using heuristic scoring.

        Uses metadata and source type to adjust scores.

        Args:
            query: The original search query.
            results: List of retrieval results to rerank.
            top_k: Number of top results to return.

        Returns:
            List of RerankResult with heuristic scores.
        """
        scored = []

        for i, result in enumerate(results):
            # Start with embedding similarity score
            score = result.score

            # Apply source-based adjustments
            if result.source == RetrievalSource.CURRICULUM:
                # Boost curriculum content slightly
                score *= 1.1
            elif result.source == RetrievalSource.EPISODIC:
                # Boost important and recent memories
                importance = result.metadata.get("importance", 0.5)
                score *= 1.0 + (importance * 0.2)
            elif result.source == RetrievalSource.ASSOCIATIVE:
                # Boost high-strength associations
                strength = result.metadata.get("strength", 0.5)
                effectiveness = result.metadata.get("times_effective", 0)
                if effectiveness > 3:
                    score *= 1.15
                score *= 1.0 + (strength * 0.1)

            # Bonus for content length (more information)
            content_length = len(result.content)
            if content_length > 100:
                score *= 1.05
            elif content_length < 30:
                score *= 0.9

            # Simple query term matching bonus
            query_terms = set(query.lower().split())
            content_terms = set(result.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                score *= 1.0 + (overlap * 0.05)

            scored.append((i, result, min(score, 1.0)))

        # Sort by score descending
        scored.sort(key=lambda x: x[2], reverse=True)

        reranked = [
            RerankResult(
                original=result,
                rerank_score=score,
                original_rank=original_rank,
                new_rank=new_rank,
            )
            for new_rank, (original_rank, result, score) in enumerate(scored[:top_k])
        ]

        logger.debug(
            "Reranked %d results with heuristics, returning top %d",
            len(results),
            len(reranked),
        )

        return reranked

    async def filter_relevant(
        self,
        query: str,
        results: list[RetrievalResult],
        min_relevance: float = 0.5,
    ) -> list[RetrievalResult]:
        """Filter results to only those above relevance threshold.

        Args:
            query: The original search query.
            results: List of retrieval results to filter.
            min_relevance: Minimum relevance score (0-1).

        Returns:
            Filtered list of relevant results.
        """
        reranked = await self.rerank(
            query=query,
            results=results,
            top_k=len(results),
        )

        return [
            r.original for r in reranked if r.rerank_score >= min_relevance
        ]

    def diversify(
        self,
        results: list[RerankResult],
        max_per_source: int = 3,
    ) -> list[RerankResult]:
        """Diversify results to include variety of sources.

        Ensures results aren't dominated by a single source.

        Args:
            results: Reranked results to diversify.
            max_per_source: Maximum results per source.

        Returns:
            Diversified list of results.
        """
        source_counts: dict[RetrievalSource, int] = {
            RetrievalSource.CURRICULUM: 0,
            RetrievalSource.EPISODIC: 0,
            RetrievalSource.ASSOCIATIVE: 0,
        }

        diversified = []
        for result in results:
            source = result.original.source
            if source_counts[source] < max_per_source:
                diversified.append(result)
                source_counts[source] += 1

        # Update new_rank
        for i, result in enumerate(diversified):
            result.new_rank = i

        return diversified

    async def explain_ranking(
        self,
        query: str,
        result: RetrievalResult,
    ) -> str:
        """Get explanation for why a result is relevant.

        Uses LLM to generate human-readable explanation.

        Args:
            query: The original search query.
            result: The result to explain.

        Returns:
            Explanation string.
        """
        if not self._llm:
            return self._explain_heuristic(query, result)

        try:
            prompt = f"""Briefly explain (1-2 sentences) why this passage is relevant to the query.

Query: {query}
Passage ({result.source.value}): {result.content}

Explanation:"""

            response = await self._llm.complete(prompt)
            return response.content.strip()

        except Exception as e:
            logger.warning("Failed to get LLM explanation: %s", str(e))
            return self._explain_heuristic(query, result)

    def _explain_heuristic(
        self,
        query: str,
        result: RetrievalResult,
    ) -> str:
        """Generate heuristic explanation for ranking.

        Args:
            query: The original search query.
            result: The result to explain.

        Returns:
            Explanation string.
        """
        explanations = []

        if result.source == RetrievalSource.CURRICULUM:
            explanations.append("Contains relevant educational content")
        elif result.source == RetrievalSource.EPISODIC:
            event_type = result.metadata.get("event_type", "event")
            explanations.append(
                f"Based on your previous {event_type} during learning"
            )
        elif result.source == RetrievalSource.ASSOCIATIVE:
            assoc_type = result.metadata.get("association_type", "interest")
            explanations.append(
                f"Connected to your {assoc_type}"
            )

        # Add score-based explanation
        if result.score >= 0.8:
            explanations.append("with high semantic similarity to your query")
        elif result.score >= 0.6:
            explanations.append("with moderate relevance to your query")

        return " ".join(explanations) + "."
