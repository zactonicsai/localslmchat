"""Shared constants and data classes for Local Context Query."""
from dataclasses import dataclass, field
from typing import Optional


TASK_QUEUE = "document-processing"
WORKFLOW_ID_PREFIX = "doc-upload"


@dataclass
class UploadInput:
    """Input for the document upload workflow."""
    doc_id: str
    filename: str
    filepath: str


@dataclass
class UploadResult:
    """Result returned by the document upload workflow."""
    doc_id: str
    filename: str
    chunks: int
    characters: int
    status: str  # "completed", "failed"
    error: Optional[str] = None


@dataclass
class ExtractResult:
    """Result of text extraction activity."""
    text: str
    characters: int


@dataclass
class ChunkResult:
    """Result of chunking activity."""
    chunks: list
    count: int


@dataclass
class EmbedStoreResult:
    """Result of embed + store activity."""
    chunks_stored: int
