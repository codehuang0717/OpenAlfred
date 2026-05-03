"""
Context Manager — Three-layer memory architecture for conversation management.

Layer 1 (Short-term): Sliding window of recent messages
Layer 2 (Mid-term): Automatic conversation summarization
Layer 3 (Long-term): Mem0 cross-session knowledge extraction

This module handles:
- Trimming old messages to stay within context window limits
- Generating summaries when conversations exceed the threshold
- Extracting user preferences/facts for long-term storage
- Injecting relevant memories into the conversation context
"""

import logging
from typing import Optional
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    RemoveMessage,
)
from core.config import config

logger = logging.getLogger("context-manager")

from logic.prompts import (
    SUMMARY_PROMPT,
    KNOWLEDGE_EXTRACTION_PROMPT,
    TITLE_GENERATION_PROMPT,
)

class ContextManager:
    """Manages the three-layer memory system for conversations."""

    def __init__(
        self,
        max_messages: int = config.MAX_CONTEXT_MESSAGES,
        summary_threshold: int = config.SUMMARY_THRESHOLD,
        max_context_tokens: int = config.MAX_CONTEXT_TOKENS,
    ):
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.max_context_tokens = max_context_tokens

    def prepare_context(
        self,
        messages: list[BaseMessage],
        conversation_summary: str = "",
        long_term_memories: str = "",
    ) -> list[BaseMessage]:
        """Prepare the message list for LLM invocation.

        Injects the conversation summary and relevant long-term memories
        as system messages, then trims to the sliding window.
        """
        context_messages: list[BaseMessage] = []

        # Inject mid-term memory (conversation summary)
        if conversation_summary:
            context_messages.append(
                SystemMessage(
                    content=f"[对话历史摘要]\n{conversation_summary}"
                )
            )

        # Inject long-term memory
        if long_term_memories and long_term_memories != "无相关记忆":
            context_messages.append(
                SystemMessage(
                    content=f"[用户长期记忆]\n{long_term_memories}"
                )
            )

        # Apply sliding window: keep only the most recent messages
        recent = messages[-self.max_messages:] if len(messages) > self.max_messages else messages
        context_messages.extend(recent)

        return context_messages

    def should_summarize(self, messages: list[BaseMessage]) -> bool:
        """Check if the conversation is long enough to trigger summarization."""
        # Count only human and AI messages (skip system/tool messages)
        content_msgs = [
            m for m in messages
            if isinstance(m, (HumanMessage, AIMessage))
        ]
        return len(content_msgs) >= self.summary_threshold

    def get_messages_to_summarize(
        self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """Split messages into older ones to summarize and recent ones to keep.

        Returns: (to_summarize, to_keep)
        """
        # Keep the most recent messages, summarize the rest
        keep_count = self.max_messages // 2  # Keep half the window
        content_msgs = [
            m for m in messages
            if isinstance(m, (HumanMessage, AIMessage))
        ]

        if len(content_msgs) <= keep_count:
            return [], messages

        # Find the index boundary in the original message list
        kept_content = 0
        split_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], (HumanMessage, AIMessage)):
                kept_content += 1
                if kept_content >= keep_count:
                    split_idx = i
                    break

        to_summarize = messages[:split_idx]
        to_keep = messages[split_idx:]
        return to_summarize, to_keep

    def format_messages_for_summary(self, messages: list[BaseMessage]) -> str:
        """Format messages into a readable string for summarization."""
        lines = []
        for m in messages:
            if isinstance(m, HumanMessage):
                lines.append(f"用户：{m.content}")
            elif isinstance(m, AIMessage):
                content = m.content or ""
                if content:
                    lines.append(f"助手：{content}")
        return "\n".join(lines)

    def build_summary_prompt(self, messages: list[BaseMessage]) -> str:
        """Build the prompt for conversation summarization."""
        conversation = self.format_messages_for_summary(messages)
        return SUMMARY_PROMPT.format(conversation=conversation)

    def build_knowledge_extraction_prompt(
        self, user_message: str, assistant_message: str
    ) -> str:
        """Build the prompt for extracting long-term knowledge."""
        return KNOWLEDGE_EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_message=assistant_message,
        )

    @staticmethod
    def build_title_prompt(first_message: str) -> str:
        """Build the prompt for auto-generating conversation titles."""
        return TITLE_GENERATION_PROMPT.format(message=first_message)
