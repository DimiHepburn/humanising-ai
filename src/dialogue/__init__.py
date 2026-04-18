"""
Dialogue sub-package
=====================

Tools for building emotionally aware conversational agents:

- `context_manager.py`     : long-horizon conversation state,
                             including emotional arcs and salient
                             memories
- `empathetic_responder.py`: an end-to-end responder that ties
                             together emotion detection, context
                             tracking and response generation

Both modules ship with lightweight default backends so the whole
pipeline is runnable without any external models, while also
exposing clean plug-in points for HuggingFace / OpenAI / local
LLM backends.
"""

from .context_manager import ConversationTurn, ConversationContext
from .empathetic_responder import (
    ResponseGenerator,
    TemplateGenerator,
    LLMGenerator,
    EmpatheticResponder,
)

__all__ = [
    "ConversationTurn",
    "ConversationContext",
    "ResponseGenerator",
    "TemplateGenerator",
    "LLMGenerator",
    "EmpatheticResponder",
]
