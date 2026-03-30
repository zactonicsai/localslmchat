"""Shared constants and data classes."""
from dataclasses import dataclass, field
from typing import Optional, List

TASK_QUEUE = "document-processing"
QUERY_TASK_QUEUE = "query-processing"
WORKFLOW_ID_PREFIX = "doc-upload"
QUERY_WORKFLOW_PREFIX = "query"
S3_RAW_PREFIX = "raw/"
S3_TEXT_PREFIX = "extracted/"
S3_QUERY_PREFIX = "queries/"
S3_ANSWER_PREFIX = "answers/"


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

@dataclass
class QueryInput:
    query_id: str
    s3_query_key: str
    model: str
    enabled_doc_ids: List[str] = field(default_factory=list)

@dataclass
class QueryResult:
    query_id: str
    status: str = "completed"
    s3_answer_key: str = ""
    error: Optional[str] = None
