# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System Explainer service for explaining EduSynapseOS.

This service provides an AI-powered agent that can explain every aspect
of EduSynapseOS to different audiences - investors, educators, technical
teams, or general users.

The agent uses RAG (Retrieval Augmented Generation) to dynamically fetch
relevant knowledge chunks based on the user's query, rather than loading
all documentation into context.

Architecture:
    1. User asks a question
    2. RAG retrieves most relevant knowledge chunks (semantic search)
    3. Only relevant chunks are sent to LLM as context
    4. LLM generates response based on retrieved context
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from src.core.config import get_settings
from src.core.config.yaml_loader import load_yaml
from src.core.intelligence.embeddings import EmbeddingService
from src.core.intelligence.llm.client import LLMClient, Message
from src.domains.system_explainer.knowledge_rag import KnowledgeRAG
from src.domains.system_explainer.schemas import (
    AudienceType,
    ConversationMessage,
    ExplainerChatRequest,
    ExplainerChatResponse,
    MessageRole,
    QuickExplainRequest,
    QuickExplainResponse,
)
from src.infrastructure.vectors import get_qdrant, init_qdrant

logger = logging.getLogger(__name__)

# In-memory session storage (for demo purposes)
# In production, this should use Redis or database
_sessions: dict[UUID, dict[str, Any]] = {}

# Available topics for quick explanations
AVAILABLE_TOPICS = {
    "elevator_pitch": {
        "title": "30-Second Elevator Pitch",
        "description": "Quick overview of what EduSynapseOS is",
    },
    "memory_system": {
        "title": "4-Layer Memory System",
        "description": "How we remember and personalize for each student",
    },
    "educational_theories": {
        "title": "7-Theory Educational Orchestrator",
        "description": "The educational science behind our personalization",
    },
    "agent_system": {
        "title": "Specialized AI Agents",
        "description": "Our multi-agent architecture for different learning contexts",
    },
    "business_model": {
        "title": "B2B SaaS Business Model",
        "description": "How we serve schools, tutoring centers, and EdTech companies",
    },
    "technical_architecture": {
        "title": "Technical Architecture",
        "description": "The AI-native tech stack powering EduSynapseOS",
    },
    "competitive_advantages": {
        "title": "Competitive Advantages",
        "description": "What makes us different from other EdTech solutions",
    },
    "lumi_companion": {
        "title": "LUMI - The Learning Companion",
        "description": "Our AI companion that builds relationships with students",
    },
}


class SystemExplainerService:
    """Service for explaining EduSynapseOS through an AI agent with RAG.

    This service uses RAG (Retrieval Augmented Generation) to dynamically
    fetch relevant knowledge based on user queries. This is more efficient
    than loading all documentation into context.

    The agent adapts its communication style based on the audience type:
    - Investors: Business focus, metrics, market opportunity
    - Educators: Pedagogical focus, theory foundations, outcomes
    - Technical: Architecture, implementation, integration
    - General: Accessible explanations, analogies, examples

    Attributes:
        _llm_client: LLM client for generating responses.
        _agent_config: Loaded agent configuration.
        _knowledge_rag: RAG retriever for knowledge base.
        _persona: Agent persona for personality.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Initialize the System Explainer service.

        Args:
            llm_client: Optional LLM client. Created if not provided.
            embedding_service: Optional embedding service. Created if not provided.
        """
        settings = get_settings()

        # Initialize LLM client
        if llm_client is None:
            llm_client = LLMClient(llm_settings=settings.llm)
        self._llm_client = llm_client

        # Initialize embedding service
        if embedding_service is None:
            embedding_service = EmbeddingService()
        self._embedding_service = embedding_service

        # Initialize Qdrant client (try to get existing, or create new)
        try:
            self._qdrant_client = get_qdrant()
        except Exception:
            # Initialize Qdrant if not already done
            init_qdrant(settings.qdrant)
            self._qdrant_client = get_qdrant()

        # Initialize Knowledge RAG
        self._knowledge_rag = KnowledgeRAG(
            qdrant_client=self._qdrant_client,
            embedding_service=self._embedding_service,
        )

        # Load agent configuration
        self._agent_config = self._load_agent_config()

        # Load persona
        self._persona = self._load_persona()

        logger.info("SystemExplainerService initialized with RAG")

    def _load_agent_config(self) -> dict[str, Any]:
        """Load the system_explainer agent configuration.

        Returns:
            Agent configuration dictionary.
        """
        config_path = Path("config/agents/system_explainer.yaml")
        try:
            config = load_yaml(config_path)
            logger.debug("Loaded agent config from %s", config_path)
            return config
        except Exception as e:
            logger.error("Failed to load agent config: %s", e)
            return {"agent": {"id": "system_explainer"}}

    def _load_persona(self) -> dict[str, Any]:
        """Load the system_explainer persona.

        Returns:
            Persona configuration dictionary.
        """
        persona_path = Path("config/personas/system_explainer.yaml")
        try:
            persona = load_yaml(persona_path)
            logger.debug("Loaded persona from %s", persona_path)
            return persona
        except Exception as e:
            logger.warning("Failed to load persona: %s", e)
            return {}

    def _build_system_prompt(
        self,
        audience: AudienceType,
        language: str = "en",
        retrieved_context: str = "",
    ) -> str:
        """Build the system prompt for the agent.

        Args:
            audience: Target audience type.
            language: Response language.
            retrieved_context: Dynamically retrieved knowledge context.

        Returns:
            Complete system prompt string.
        """
        agent_config = self._agent_config.get("agent", {})
        system_prompt_config = agent_config.get("system_prompt", {})

        # Get role definition
        role = system_prompt_config.get("role", "You are SYNAPSE, an expert on EduSynapseOS.")

        # Get knowledge sections (static, brief overview)
        knowledge_sections = system_prompt_config.get("knowledge_sections", [])
        knowledge_text = ""
        for section in knowledge_sections:
            section_title = section.get("title", "")
            section_content = section.get("content", "")
            knowledge_text += f"\n\n## {section_title}\n{section_content}"

        # Get audience-specific approach
        audience_approaches = system_prompt_config.get("audience_approaches", [])
        audience_approach = ""
        for approach in audience_approaches:
            if approach.get("audience") == audience.value or (
                audience == AudienceType.AUTO and approach.get("audience") == "general"
            ):
                audience_approach = approach.get("approach", "")
                break

        # Get rules
        rules = system_prompt_config.get("rules", [])
        rules_text = ""
        for rule in rules:
            rules_text += f"\n- **{rule.get('title', '')}**: {rule.get('content', '')}"

        # Get personality
        personality = system_prompt_config.get("personality", "")

        # Build the full prompt with RAG context
        prompt = f"""# SYNAPSE - EduSynapseOS Expert Agent

{role}

## CURRENT CONTEXT
- Audience: {audience.value}
- Language: {language}

## AUDIENCE ADAPTATION
{audience_approach}

## CORE KNOWLEDGE (Overview)
{knowledge_text}

## RELEVANT KNOWLEDGE (Retrieved for this query)
The following information was retrieved from the knowledge base based on the user's question.
Use this to provide accurate, detailed answers:

<retrieved_knowledge>
{retrieved_context if retrieved_context else "No specific knowledge retrieved. Use your general understanding of EduSynapseOS."}
</retrieved_knowledge>

## RULES
{rules_text}

## PERSONALITY
{personality}

## RESPONSE GUIDELINES
1. Be articulate, confident, and passionate about the technology
2. Use vivid analogies to explain complex concepts
3. Adapt depth and terminology to the audience
4. Base your answers primarily on the retrieved knowledge above
5. If the retrieved knowledge doesn't contain the answer, say so honestly
6. Respond in {language}
7. If asked about pricing or specific numbers you don't know, say you'd be happy to connect them with the team

## FORMATTING REQUIREMENTS (VERY IMPORTANT)
- **Use short paragraphs**: Break your response into 2-3 sentence paragraphs with blank lines between them
- **Use bullet points or numbered lists** when explaining multiple features, benefits, or steps
- **Use bold** for key terms and important concepts
- **Structure your response** with clear sections when answering complex questions
- **Keep responses scannable**: Users should be able to quickly scan and understand the main points
- **Avoid wall of text**: Never write more than 3 sentences in a single paragraph
- Example structure:
  [Brief intro paragraph]

  [Key point 1 with details]

  [Key point 2 with details]

  [Closing with call to action or follow-up question]
"""

        return prompt

    def _get_or_create_session(self, session_id: UUID | None) -> tuple[UUID, list[ConversationMessage]]:
        """Get existing session or create a new one.

        Args:
            session_id: Optional existing session ID.

        Returns:
            Tuple of (session_id, conversation_history).
        """
        if session_id and session_id in _sessions:
            session = _sessions[session_id]
            session["last_activity"] = datetime.now(timezone.utc)
            return session_id, session["messages"]

        # Create new session
        new_id = uuid4()
        _sessions[new_id] = {
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "messages": [],
            "audience": AudienceType.AUTO,
        }
        return new_id, []

    def _detect_audience(self, message: str, history: list[ConversationMessage]) -> AudienceType:
        """Detect the audience type from message content.

        Args:
            message: Current user message.
            history: Conversation history.

        Returns:
            Detected audience type.
        """
        message_lower = message.lower()

        # Technical indicators
        tech_keywords = ["api", "architecture", "database", "code", "integration", "sdk", "endpoint", "deploy"]
        if any(kw in message_lower for kw in tech_keywords):
            return AudienceType.TECHNICAL

        # Investor indicators
        investor_keywords = ["roi", "market", "revenue", "investment", "funding", "valuation", "traction", "metrics", "moat"]
        if any(kw in message_lower for kw in investor_keywords):
            return AudienceType.INVESTOR

        # Educator indicators
        educator_keywords = ["pedagogy", "curriculum", "classroom", "teaching", "student outcomes", "learning theory"]
        if any(kw in message_lower for kw in educator_keywords):
            return AudienceType.EDUCATOR

        return AudienceType.GENERAL

    def _suggest_topics(
        self,
        message: str,
        response: str,
        retrieved_chunks: list | None = None,
    ) -> list[str]:
        """Suggest related topics based on conversation and retrieved chunks.

        Uses RAG chunk sources to intelligently suggest related topics
        that the user might want to explore next.

        Args:
            message: User's message.
            response: Agent's response.
            retrieved_chunks: Chunks retrieved from RAG (optional).

        Returns:
            List of suggested topic IDs (max 4).
        """
        # Map document names to topic IDs
        doc_to_topic = {
            "02-MEMORY-SYSTEM.md": "memory_system",
            "03-EDUCATIONAL-THEORIES.md": "educational_theories",
            "04-AI-AGENTS.md": "agent_system",
            "05-WORKFLOWS.md": "agent_system",
            "08-EMOTIONAL-INTELLIGENCE.md": "lumi_companion",
            "11-TECHNICAL-STACK.md": "technical_architecture",
            "12-BUSINESS-MODEL.md": "business_model",
            "14-COMPETITIVE-ANALYSIS.md": "competitive_advantages",
            "01-EXECUTIVE-SUMMARY.md": "elevator_pitch",
        }

        # Collect topics from retrieved chunks
        chunk_topics: set[str] = set()
        if retrieved_chunks:
            for chunk in retrieved_chunks:
                doc_name = chunk.chunk.document if hasattr(chunk, "chunk") else ""
                if doc_name in doc_to_topic:
                    chunk_topics.add(doc_to_topic[doc_name])

        # Also use keyword matching for additional suggestions
        message_lower = message.lower()
        keyword_topics: list[str] = []

        keyword_map = {
            "memory_system": ["memory", "remember", "hafıza", "episodic", "semantic"],
            "educational_theories": ["theory", "theories", "pedagogy", "bloom", "vygotsky", "zpd", "socratic"],
            "agent_system": ["agent", "tutor", "assessor", "practice", "workflow"],
            "lumi_companion": ["lumi", "companion", "emotional", "friend", "support"],
            "business_model": ["business", "price", "pricing", "revenue", "subscription", "b2b"],
            "technical_architecture": ["technical", "architecture", "stack", "database", "api"],
            "competitive_advantages": ["competitor", "different", "advantage", "moat", "khan", "duolingo"],
            "elevator_pitch": ["what is", "overview", "summary", "explain", "nedir"],
        }

        for topic_id, keywords in keyword_map.items():
            if any(kw in message_lower for kw in keywords):
                keyword_topics.append(topic_id)

        # Combine chunk-based and keyword-based suggestions
        # Prioritize chunk-based (more relevant) but add keyword-based too
        all_topics = list(chunk_topics) + [t for t in keyword_topics if t not in chunk_topics]

        # Exclude topics that seem to be the main focus of current question
        # (suggest related but different topics)
        main_topic = None
        for topic_id, keywords in keyword_map.items():
            if sum(1 for kw in keywords if kw in message_lower) >= 2:
                main_topic = topic_id
                break

        if main_topic and main_topic in all_topics:
            all_topics.remove(main_topic)

        # Add default suggestions if we don't have enough
        default_topics = ["memory_system", "educational_theories", "agent_system", "business_model"]
        for dt in default_topics:
            if dt not in all_topics and dt != main_topic:
                all_topics.append(dt)
            if len(all_topics) >= 4:
                break

        return all_topics[:4]

    def _build_rag_query(self, message: str, history: list[ConversationMessage]) -> str:
        """Build an optimized query for RAG retrieval.

        Combines the current message with recent conversation context
        to improve retrieval relevance.

        Args:
            message: Current user message.
            history: Conversation history.

        Returns:
            Optimized query string for RAG.
        """
        # Start with the current message
        query_parts = [message]

        # Add recent context if available (last 2 exchanges)
        if history:
            recent_messages = history[-4:]  # Last 2 user + 2 assistant messages
            for msg in recent_messages:
                if msg.role == MessageRole.USER:
                    query_parts.append(msg.content)

        # Combine and limit length
        combined = " ".join(query_parts)
        if len(combined) > 500:
            combined = combined[:500]

        return combined

    async def chat(self, request: ExplainerChatRequest) -> ExplainerChatResponse:
        """Process a chat message and generate a response using RAG.

        Args:
            request: Chat request with message and optional session.

        Returns:
            Chat response with agent's reply.
        """
        # Get or create session
        session_id, history = self._get_or_create_session(request.session_id)

        # Detect or use provided audience
        audience = request.audience
        if audience == AudienceType.AUTO:
            audience = self._detect_audience(request.message, history)

        # Update session audience
        _sessions[session_id]["audience"] = audience

        # Build RAG query (combines current message with context)
        rag_query = self._build_rag_query(request.message, history)

        # Retrieve relevant knowledge chunks
        retrieved_chunks = await self._knowledge_rag.retrieve(
            query=rag_query,
            limit=7,  # Get top 7 most relevant chunks
            min_score=0.25,  # Lower threshold for broader coverage
        )

        # Format retrieved context
        retrieved_context = self._knowledge_rag.format_context(
            retrieved_chunks,
            max_length=8000,
        )

        logger.debug(
            "Retrieved %d chunks for query (scores: %s)",
            len(retrieved_chunks),
            [f"{c.score:.2f}" for c in retrieved_chunks],
        )

        # Build system prompt with retrieved context
        system_prompt = self._build_system_prompt(
            audience=audience,
            language=request.language,
            retrieved_context=retrieved_context,
        )

        # Build conversation messages for LLM
        messages = [{"role": "system", "content": system_prompt}]

        # Add history (last 10 messages for context)
        for msg in history[-10:]:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": request.message})

        # Generate response
        try:
            llm_response = await self._llm_client.complete_with_messages(
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            response_text = llm_response.content
        except Exception as e:
            logger.error("LLM error: %s", e)
            response_text = (
                "I apologize, but I'm having trouble generating a response right now. "
                "Please try again, or feel free to ask about a specific topic like our "
                "memory system, educational theories, or business model."
            )

        # Update session history
        history.append(ConversationMessage(role=MessageRole.USER, content=request.message))
        history.append(ConversationMessage(role=MessageRole.ASSISTANT, content=response_text))
        _sessions[session_id]["messages"] = history

        # Suggest related topics based on message and retrieved chunks
        suggested = self._suggest_topics(request.message, response_text, retrieved_chunks)

        return ExplainerChatResponse(
            message=response_text,
            session_id=session_id,
            audience_detected=audience,
            suggested_topics=suggested,
            metadata={
                "message_count": len(history),
                "language": request.language,
                "chunks_retrieved": len(retrieved_chunks),
                "top_chunk_score": retrieved_chunks[0].score if retrieved_chunks else 0,
            },
        )

    async def chat_stream(
        self,
        request: ExplainerChatRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat response token by token.

        Yields SSE events with tokens as they are generated.
        Final event includes metadata and suggested topics.

        Args:
            request: Chat request with message and optional session ID.

        Yields:
            SSE event dictionaries with type and data.
        """
        # Get or create session
        session_id, history = self._get_or_create_session(request.session_id)

        # Detect or use provided audience
        audience = request.audience
        if audience == AudienceType.AUTO:
            audience = self._detect_audience(request.message, history)

        # Update session audience
        _sessions[session_id]["audience"] = audience

        # Send initial event with session info
        yield {
            "event": "start",
            "data": {
                "session_id": str(session_id),
                "audience_detected": audience.value,
            },
        }

        # Build RAG query
        rag_query = self._build_rag_query(request.message, history)

        # Retrieve relevant knowledge chunks
        retrieved_chunks = await self._knowledge_rag.retrieve(
            query=rag_query,
            limit=7,
            min_score=0.25,
        )

        # Format retrieved context
        retrieved_context = self._knowledge_rag.format_context(
            retrieved_chunks,
            max_length=8000,
        )

        # Build system prompt with retrieved context
        system_prompt = self._build_system_prompt(
            audience=audience,
            language=request.language,
            retrieved_context=retrieved_context,
        )

        # Build conversation messages for LLM
        messages = [{"role": "system", "content": system_prompt}]

        # Add history (last 10 messages for context)
        for msg in history[-10:]:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": request.message})

        # Stream response tokens
        full_response = ""
        try:
            async for token in self._llm_client.stream(
                prompt=request.message,
                system_prompt=system_prompt,
                messages=[
                    Message(role=msg["role"], content=msg["content"])
                    for msg in messages[1:-1]  # Skip system (already in system_prompt) and last user msg
                ],
                temperature=0.7,
                max_tokens=2048,
            ):
                full_response += token
                yield {
                    "event": "token",
                    "data": {"token": token},
                }

        except Exception as e:
            logger.error("LLM streaming error: %s", e)
            error_msg = "I apologize, but I'm having trouble generating a response."
            yield {
                "event": "error",
                "data": {"error": error_msg},
            }
            full_response = error_msg

        # Update session history
        history.append(ConversationMessage(role=MessageRole.USER, content=request.message))
        history.append(ConversationMessage(role=MessageRole.ASSISTANT, content=full_response))
        _sessions[session_id]["messages"] = history

        # Suggest related topics
        suggested = self._suggest_topics(request.message, full_response, retrieved_chunks)

        # Send completion event with metadata
        yield {
            "event": "done",
            "data": {
                "session_id": str(session_id),
                "audience_detected": audience.value,
                "suggested_topics": suggested,
                "metadata": {
                    "message_count": len(history),
                    "language": request.language,
                    "chunks_retrieved": len(retrieved_chunks),
                    "top_chunk_score": retrieved_chunks[0].score if retrieved_chunks else 0,
                },
            },
        }

    async def quick_explain(self, request: QuickExplainRequest) -> QuickExplainResponse:
        """Get a quick explanation of a specific topic using RAG.

        This is a one-shot explanation without conversation context,
        useful for tooltips, info panels, etc.

        Args:
            request: Request with topic and audience.

        Returns:
            Quick explanation response.
        """
        topic_info = AVAILABLE_TOPICS.get(request.topic)
        if not topic_info:
            return QuickExplainResponse(
                topic=request.topic,
                explanation=f"Topic '{request.topic}' not found. Available topics: {', '.join(AVAILABLE_TOPICS.keys())}",
                related_topics=list(AVAILABLE_TOPICS.keys())[:5],
            )

        # Build query from topic info
        rag_query = f"{topic_info['title']} {topic_info['description']}"

        # Retrieve relevant knowledge
        retrieved_chunks = await self._knowledge_rag.retrieve(
            query=rag_query,
            limit=5,
            min_score=0.3,
        )

        retrieved_context = self._knowledge_rag.format_context(
            retrieved_chunks,
            max_length=6000,
        )

        # Build a focused prompt for this topic
        system_prompt = self._build_system_prompt(
            audience=request.audience,
            language=request.language,
            retrieved_context=retrieved_context,
        )

        user_prompt = f"""Please provide a clear, engaging explanation of: {topic_info['title']}

Focus on: {topic_info['description']}

Target audience: {request.audience.value}
Language: {request.language}

Provide a comprehensive but focused explanation (2-4 paragraphs). Use analogies where helpful."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            llm_response = await self._llm_client.complete_with_messages(
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            explanation = llm_response.content
        except Exception as e:
            logger.error("LLM error in quick_explain: %s", e)
            explanation = f"Unable to generate explanation for {topic_info['title']}. Please try again."

        # Get related topics
        all_topics = list(AVAILABLE_TOPICS.keys())
        related = [t for t in all_topics if t != request.topic][:3]

        return QuickExplainResponse(
            topic=request.topic,
            explanation=explanation,
            related_topics=related,
        )

    async def index_knowledge_base(self, force_reindex: bool = False) -> int:
        """Index the knowledge base documents into Qdrant.

        This should be called once when setting up the system,
        or when knowledge base documents are updated.

        Args:
            force_reindex: If True, delete existing index and recreate.

        Returns:
            Number of chunks indexed.
        """
        return await self._knowledge_rag.index_knowledge_base(force_reindex=force_reindex)

    async def is_knowledge_indexed(self) -> bool:
        """Check if knowledge base is indexed.

        Returns:
            True if knowledge base is indexed and ready.
        """
        return await self._knowledge_rag.is_indexed()

    def get_available_topics(self) -> dict[str, dict[str, str]]:
        """Get list of available topics for quick explanations.

        Returns:
            Dictionary of topic IDs to their info.
        """
        return AVAILABLE_TOPICS

    def get_session_info(self, session_id: UUID) -> dict[str, Any] | None:
        """Get information about a session.

        Args:
            session_id: Session identifier.

        Returns:
            Session info dict or None if not found.
        """
        session = _sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "message_count": len(session["messages"]),
            "audience": session["audience"].value,
        }

    def clear_session(self, session_id: UUID) -> bool:
        """Clear a session.

        Args:
            session_id: Session to clear.

        Returns:
            True if session was cleared, False if not found.
        """
        if session_id in _sessions:
            del _sessions[session_id]
            return True
        return False


# Singleton instance for the service
_service_instance: SystemExplainerService | None = None


def get_system_explainer_service() -> SystemExplainerService:
    """Get the singleton SystemExplainerService instance.

    Returns:
        SystemExplainerService instance.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = SystemExplainerService()
    return _service_instance
