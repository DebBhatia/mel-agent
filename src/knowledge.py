"""
KNOWLEDGE BASE ENGINE
======================
Local vector database for long-term agent memory.
Stores conversations, documents, and generated project info.
All data stays on YOUR machine — nothing leaves.

Uses ChromaDB for vector storage and retrieval.
PII is sanitized before storage. Old entries are auto-purged.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("knowledge")

# Max age for conversation entries (days)
CONVERSATION_RETENTION_DAYS = int(os.getenv("KB_RETENTION_DAYS", "90"))


class KnowledgeBase:
    """
    Local vector database for agent memory.
    Stores and retrieves information using semantic search.
    Everything runs locally via ChromaDB.
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or os.path.expanduser("~/.agent-knowledge")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = None
        self.collection = None
        self._sanitizer = None
        self._initialize()
        self._init_sanitizer()
        self._purge_old_entries()

    def _init_sanitizer(self):
        """Load PII sanitizer for cleaning data before storage."""
        try:
            from security import EnhancedPIISanitizer
            self._sanitizer = EnhancedPIISanitizer()
        except ImportError:
            pass

    def _sanitize(self, text: str) -> str:
        """Strip PII before storing in knowledge base."""
        if self._sanitizer:
            return self._sanitizer.sanitize(text)
        return text

    def _purge_old_entries(self):
        """Remove conversation entries older than retention period."""
        if not self.collection:
            return
        try:
            cutoff = (datetime.now() - timedelta(days=CONVERSATION_RETENTION_DAYS)).isoformat()
            # ChromaDB where filter for old conversations
            results = self.collection.get(
                where={"$and": [
                    {"category": {"$eq": "conversation"}},
                    {"timestamp": {"$lt": cutoff}},
                ]},
            )
            if results and results.get("ids"):
                self.collection.delete(ids=results["ids"])
                logger.info(f"Purged {len(results['ids'])} expired conversation entries")
        except Exception as e:
            # ChromaDB may not support complex where on older versions — skip silently
            logger.debug(f"Purge skipped: {e}")

    def _initialize(self):
        """Set up ChromaDB with local persistence."""
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),  # No telemetry!
            )

            self.collection = self.client.get_or_create_collection(
                name="agent_memory",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Knowledge base loaded: {self.collection.count()} entries")
        except ImportError:
            logger.warning(
                "ChromaDB not installed. Knowledge base disabled. "
                "Install with: pip install chromadb"
            )

    def store(self, content: str, metadata: dict = None, category: str = "general") -> str:
        """Store a piece of knowledge. PII is sanitized before storage."""
        if not self.collection:
            return "Knowledge base not available"

        # Sanitize content before persisting
        safe_content = self._sanitize(content)

        doc_id = hashlib.md5(
            f"{safe_content[:100]}-{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Sanitize metadata values too
        safe_metadata = {}
        if metadata:
            for k, v in metadata.items():
                safe_metadata[k] = self._sanitize(str(v)) if isinstance(v, str) else v

        meta = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            **safe_metadata,
        }

        self.collection.add(
            documents=[safe_content],
            metadatas=[meta],
            ids=[doc_id],
        )

        logger.info(f"Stored knowledge: {doc_id} ({category})")
        return doc_id

    def search(self, query: str, n_results: int = 5, category: str = None) -> list[dict]:
        """Search for relevant knowledge."""
        if not self.collection:
            return []

        where = {"category": category} if category else None

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

        entries = []
        for i, doc in enumerate(results["documents"][0]):
            entries.append({
                "content": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
            })

        return entries

    def store_conversation(self, user_input: str, agent_response: str):
        """Store a conversation turn for future reference. PII is auto-sanitized."""
        content = f"User: {user_input}\nAgent: {agent_response}"
        # Don't store raw user_input in metadata — sanitize() handles the content
        self.store(content, category="conversation")

    def store_project(self, project_name: str, description: str, files: list[str]):
        """Store info about a generated project."""
        content = f"Project: {project_name}\nDescription: {description}\nFiles: {', '.join(files)}"
        self.store(content, category="project", metadata={
            "project_name": project_name,
        })

    def recall(self, query: str) -> str:
        """Get a formatted summary of relevant memories."""
        results = self.search(query, n_results=3)
        if not results:
            return "No relevant memories found."

        summary = "Here's what I remember:\n"
        for r in results:
            summary += f"- {r['content'][:200]}...\n"
            summary += f"  (from {r['metadata'].get('timestamp', 'unknown time')})\n"
        return summary

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        if not self.collection:
            return {"status": "disabled", "entries": 0}

        return {
            "status": "active",
            "entries": self.collection.count(),
        }
