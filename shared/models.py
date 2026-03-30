"""Shared constants and data classes."""
from dataclasses import dataclass, field
from typing import Optional, List

TASK_QUEUE = "document-processing"
WORKFLOW_ID_PREFIX = "doc-upload"
S3_RAW_PREFIX = "raw/"
S3_TEXT_PREFIX = "extracted/"


@dataclass
class UploadInput:
    doc_id: str
    filename: str
    s3_raw_key: str


@dataclass
class UploadResult:
    doc_id: str
    filename: str
    chunks: int
    characters: int
    status: str
    error: Optional[str] = None


@dataclass
class ExtractResult:
    s3_text_key: str
    characters: int


@dataclass
class ChunkResult:
    chunks: List[str] = field(default_factory=list)
    count: int = 0


@dataclass
class EmbedStoreResult:
    chunks_stored: int = 0
