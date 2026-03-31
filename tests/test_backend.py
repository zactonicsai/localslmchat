"""Tests for Local Context Query — async query via S3 + Temporal + WebSocket."""
import io, os, sys, json, tempfile
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock
import pytest, pytest_asyncio
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
sys.path.insert(0, PROJECT_ROOT)
os.environ.update({"CHROMA_HOST":"localhost","CHROMA_PORT":"8000","OLLAMA_BASE_URL":"http://localhost:11434",
    "TEMPORAL_ADDRESS":"localhost:7233","EMBED_MODEL":"nomic-embed-text","S3_ENDPOINT":"http://localhost:4566",
    "S3_BUCKET":"lcq-documents","AWS_ACCESS_KEY_ID":"test","AWS_SECRET_ACCESS_KEY":"test","AWS_DEFAULT_REGION":"us-east-1"})

class FakeS3:
    def __init__(self): self._o={}
    def put_object(self,Bucket,Key,Body): self._o[f"{Bucket}/{Key}"]=Body if isinstance(Body,bytes) else Body.encode()
    def get_object(self,Bucket,Key):
        k=f"{Bucket}/{Key}"
        if k not in self._o: raise Exception(f"NoSuchKey: {Key}")
        return {"Body":io.BytesIO(self._o[k])}
    def head_bucket(self,Bucket): return {}
    def list_objects_v2(self,Bucket,Prefix=""): return {"Contents":[{"Key":k.split("/",1)[1]} for k in self._o if k.startswith(f"{Bucket}/{Prefix}")]}
    def delete_object(self,Bucket,Key): self._o.pop(f"{Bucket}/{Key}",None)
fake_s3=FakeS3()

class FakeCollection:
    def __init__(self): self._ids,self._emb,self._meta,self._docs=[],[],[],[]
    def count(self): return len(self._ids)
    def add(self,ids,embeddings,metadatas,documents): self._ids.extend(ids);self._emb.extend(embeddings);self._meta.extend(metadatas);self._docs.extend(documents)
    def get(self,include=None): return {"ids":list(self._ids),"metadatas":list(self._meta),"documents":list(self._docs)}
    def query(self,query_embeddings,n_results=5,include=None):
        n=min(n_results,len(self._ids));return{"ids":[self._ids[:n]],"documents":[self._docs[:n]],"metadatas":[self._meta[:n]],"distances":[[.1*i for i in range(n)]]}
    def delete(self,ids):
        idx=[i for i,x in enumerate(self._ids) if x in ids]
        for i in sorted(idx,reverse=True): self._ids.pop(i);self._emb.pop(i);self._meta.pop(i);self._docs.pop(i)
fake_collection=FakeCollection()

FAKE_EMBED=[.1]*768
async def mock_ollama_get(path,timeout=10.0):
    if"/api/tags"in path:return{"models":[{"name":"qwen3.5:9b","size":2e9},{"name":"nomic-embed-text:latest","size":5e8}]}
    return{}

_wf={}
@dataclass
class FakeHandle:
    wf_id:str
    async def describe(self):
        if self.wf_id not in _wf:raise RuntimeError(f"not found: {self.wf_id}")
        m=MagicMock();m.status=MagicMock();m.status.name="COMPLETED" if _wf[self.wf_id].get("result") else "RUNNING";return m
    async def result(self):
        r=_wf.get(self.wf_id,{}).get("result")
        if r:return r;raise RuntimeError("Not done")
class FakeTemporal:
    async def start_workflow(self,wf_name,inp,id,task_queue): _wf[id]={"input":inp,"result":None}
    def get_workflow_handle(self,wf_id): return FakeHandle(wf_id)
_fake_t=FakeTemporal()

def _sim_upload(doc_id,filename,chunks):
    from datetime import datetime,timezone
    _wf[f"doc-upload-{doc_id}"]={"result":{"doc_id":doc_id,"filename":filename,"chunks":chunks,"characters":1000,"status":"completed","error":None}}
    now=datetime.now(timezone.utc).isoformat()
    for i in range(chunks):
        fake_collection._ids.append(f"{doc_id}_chunk_{i}");fake_collection._emb.append(FAKE_EMBED)
        fake_collection._meta.append({"doc_id":doc_id,"filename":filename,"chunk_index":i,"total_chunks":chunks,"uploaded_at":now})
        fake_collection._docs.append(f"Chunk {i} from {filename}. Data here.")

def _sim_query_answer(query_id):
    """Simulate worker writing answer to S3."""
    key = f"lcq-documents/answers/{query_id}.json"
    fake_s3._o[key] = json.dumps({"query_id": query_id, "answer": "The answer is 42.", "sources": [{"filename": "test.txt", "doc_id": "abc", "chunk_index": 0, "distance": 0.1}]}).encode()

@pytest_asyncio.fixture(autouse=True)
async def reset():
    fake_collection._ids.clear();fake_collection._emb.clear();fake_collection._meta.clear();fake_collection._docs.clear()
    fake_s3._o.clear();_wf.clear()
    import main;main._temporal=None;main._chroma_client=None;main._collection=None
    yield

@pytest_asyncio.fixture
async def client():
    import main
    main.get_collection=lambda:fake_collection;main.get_temporal=AsyncMock(return_value=_fake_t)
    main._s3=lambda:fake_s3;main.ollama_get=mock_ollama_get
    transport=ASGITransport(app=main.app)
    async with AsyncClient(transport=transport,base_url="http://test") as c: yield c

@pytest.mark.asyncio
async def test_health(client):
    r=await client.get("/api/health");assert r.status_code==200

@pytest.mark.asyncio
async def test_models(client):
    r=await client.get("/api/models");assert "qwen3.5:9b" in [m["name"] for m in r.json()["models"]]

@pytest.mark.asyncio
async def test_upload_to_s3(client):
    r=await client.post("/api/upload",files={"file":("test.txt",io.BytesIO(b"Hello world"),"text/plain")})
    assert r.status_code==200;d=r.json();assert d["status"]=="processing"
    assert f"lcq-documents/raw/{d['doc_id']}/test.txt" in fake_s3._o

@pytest.mark.asyncio
async def test_upload_empty_400(client):
    r=await client.post("/api/upload",files={"file":("e.txt",io.BytesIO(b""),"text/plain")});assert r.status_code==400

@pytest.mark.asyncio
async def test_upload_status_processing(client):
    r=await client.post("/api/upload",files={"file":("d.txt",io.BytesIO(b"Data"),"text/plain")})
    s=await client.get(f"/api/upload/{r.json()['doc_id']}/status");assert s.json()["status"]=="processing"

@pytest.mark.asyncio
async def test_upload_status_completed(client):
    r=await client.post("/api/upload",files={"file":("r.txt",io.BytesIO(b"Report"),"text/plain")})
    did=r.json()["doc_id"];_sim_upload(did,"r.txt",3)
    s=await client.get(f"/api/upload/{did}/status");assert s.json()["status"]=="completed";assert s.json()["chunks"]==3

@pytest.mark.asyncio
async def test_list_documents(client):
    _sim_upload("a","notes.txt",2);_sim_upload("b","guide.txt",5)
    r=await client.get("/api/documents");assert len(r.json()["documents"])==2

@pytest.mark.asyncio
async def test_query_returns_immediately(client):
    """POST /api/query returns query_id + queued status immediately."""
    r=await client.post("/api/query",json={"query":"What is 42?","model":"qwen3.5:9b"})
    assert r.status_code==200;d=r.json();assert d["status"]=="queued";assert d["query_id"]
    # Question saved to S3
    assert f"lcq-documents/queries/{d['query_id']}.json" in fake_s3._o

@pytest.mark.asyncio
async def test_query_answer_not_ready(client):
    """GET /api/query/{id}/answer returns processing when answer not on S3 yet."""
    r=await client.post("/api/query",json={"query":"Test","model":"qwen3.5:9b"})
    qid=r.json()["query_id"]
    a=await client.get(f"/api/query/{qid}/answer");assert a.json()["status"]=="processing"

@pytest.mark.asyncio
async def test_query_answer_ready(client):
    """GET /api/query/{id}/answer returns answer when S3 has it."""
    r=await client.post("/api/query",json={"query":"Test","model":"qwen3.5:9b"})
    qid=r.json()["query_id"]
    _sim_query_answer(qid)
    a=await client.get(f"/api/query/{qid}/answer")
    assert a.json()["status"]=="completed";assert "42" in a.json()["answer"];assert len(a.json()["sources"])>0

@pytest.mark.asyncio
async def test_ws_callback_broadcasts(client):
    """POST /api/internal/query-complete reads answer from S3."""
    qid="test123";_sim_query_answer(qid)
    r=await client.post("/api/internal/query-complete",json={"query_id":qid})
    assert r.status_code==200;assert r.json()["ok"]==True

@pytest.mark.asyncio
async def test_delete(client):
    _sim_upload("d1","tmp.txt",2);assert fake_collection.count()==2
    r=await client.delete("/api/documents/d1");assert r.status_code==200;assert fake_collection.count()==0

@pytest.mark.asyncio
async def test_full_flow(client):
    # Empty → query question saved but no answer yet
    r1=await client.post("/api/query",json={"query":"Hi?","model":"qwen3.5:9b"})
    qid=r1.json()["query_id"]
    a1=await client.get(f"/api/query/{qid}/answer");assert a1.json()["status"]=="processing"
    # Upload + simulate complete
    up=await client.post("/api/upload",files={"file":("guide.txt",io.BytesIO(b"Python guide "*50),"text/plain")})
    did=up.json()["doc_id"];_sim_upload(did,"guide.txt",4)
    # Docs visible
    docs=(await client.get("/api/documents")).json()["documents"];assert len(docs)==1
    # New query → simulate worker answer
    r2=await client.post("/api/query",json={"query":"What is Python?","model":"qwen3.5:9b"})
    qid2=r2.json()["query_id"];_sim_query_answer(qid2)
    a2=await client.get(f"/api/query/{qid2}/answer");assert a2.json()["status"]=="completed"
    # Delete
    await client.delete(f"/api/documents/{did}");assert fake_collection.count()==0

@pytest.mark.asyncio
async def test_all_json(client):
    for url in ["/api/health","/api/models","/api/documents"]:
        r=await client.get(url);assert "application/json" in r.headers.get("content-type","")
