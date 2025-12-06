import os
import json
import time
import pickle
from typing import List, Tuple, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in env before starting server")

client = OpenAI(api_key=OPENAI_API_KEY)
APP_PORT = int(os.getenv("APP_PORT", "8000"))
MODEL_CHAT = os.getenv("MODEL_CHAT", "gpt-4o")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
INDEX_PKL = os.getenv("INDEX_PKL", "rag_index.pkl")
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))

# Optional FAISS
try:
    import faiss

    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

app = FastAPI(title="RAG-enabled Medical Triage Server")

# ---------- Data structures ----------
index = {"embeddings": None, "metadatas": [], "dim": None}
faiss_index = None


# ---------- Pydantic models ----------
class DirectIngestRequest(BaseModel):
    title: str
    content: str
    category: str = "Medical Knowledge"
    source: str = "Direct Input"


class IngestRequest(BaseModel):
    url: str


class TriageRequest(BaseModel):
    history: str
    max_questions: int = 5


class BuildIndexRequest(BaseModel):
    persist: bool = True


# ---------- Trusted medical sites ----------
TRUSTED_SITES = [
    {"url": "https://www.cdc.gov/health/conditions", "category": "Government Health"},
    {"url": "https://www.who.int/health-topics", "category": "International Health"},
    {"url": "https://www.nhs.uk/conditions", "category": "UK Health Service"},
    {"url": "https://www.mayoclinic.org/diseases-conditions", "category": "Medical Research"},
    {"url": "https://www.medicinenet.com/conditions.htm", "category": "Medical Reference"},
]


# ---------- Utilities ----------
def scrape_url(url: str) -> Tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }
    resp = requests.get(url, timeout=15, headers=headers)
    if resp.status_code == 403:
        raise Exception("403 Forbidden")
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    content_div = soup.find("div", {"id": "content"}) or soup.find("main") or soup.body
    if content_div:
        paragraphs = [p.get_text(separator=" ", strip=True) for p in content_div.find_all("p")]
    else:
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    text = "\n\n".join([p for p in paragraphs if len(p) > 50])
    if not text:
        return title, "No content extracted."
    return title, text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    i = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        res = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([r.embedding for r in res.data])
    return np.array(embeddings, dtype=np.float32)


def faiss_index_to_bytes() -> Optional[bytes]:
    if not _HAS_FAISS or faiss_index is None:
        return None
    return faiss.serialize_index(faiss_index)


def faiss_index_from_bytes(b: Optional[bytes]):
    if not _HAS_FAISS or b is None:
        return None
    return faiss.deserialize_index(b)


def build_faiss_from_numpy(embs: np.ndarray):
    global faiss_index
    if _HAS_FAISS:
        d = embs.shape[1]
        idx = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embs)
        idx.add(embs)
        faiss_index = idx


def save_index(path: str = INDEX_PKL) -> str:
    with open(path, "wb") as f:
        pickle.dump({
            "index": index,
            "has_faiss": _HAS_FAISS,
            "faiss_index": faiss_index_to_bytes()
        }, f)
    return path


def load_index(path: str = INDEX_PKL) -> bool:
    global index, faiss_index
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        data = pickle.load(f)
    index.update(data.get("index", {}))
    if data.get("has_faiss") and data.get("faiss_index") is not None:
        faiss_index = faiss_index_from_bytes(data["faiss_index"])
    return True


def add_to_index(new_embeddings: np.ndarray, new_metadatas: List[dict]):
    global index
    if index["embeddings"] is None:
        index["embeddings"] = new_embeddings
    else:
        index["embeddings"] = np.vstack([index["embeddings"], new_embeddings])
    index["metadatas"].extend(new_metadatas)
    index["dim"] = index["embeddings"].shape[1]
    build_faiss_from_numpy(index["embeddings"])


def retrieve(query: str, top_k: int = TOP_K) -> List[dict]:
    if index["embeddings"] is None or len(index["metadatas"]) == 0:
        return []
    enhanced_query = query + " symptoms causes treatment medical condition"
    try:
        q_emb = embed_texts([enhanced_query])[0]
    except Exception:
        return []

    if _HAS_FAISS and faiss_index is not None:
        q = q_emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        search_k = min(top_k * 3, len(index["metadatas"]))
        distances, idxs = faiss_index.search(q, search_k)
        res: List[dict] = []
        for i, idx in enumerate(idxs[0]):
            if idx < 0 or idx >= len(index["metadatas"]):
                continue
            meta = index["metadatas"][idx].copy()
            meta["score"] = float(distances[0][i])
            if meta["score"] >= SIMILARITY_THRESHOLD:
                res.append(meta)
        return sorted(res, key=lambda x: x["score"], reverse=True)[:top_k]

    embs = index["embeddings"]
    scores = embs.dot(q_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-12)
    search_k = min(top_k * 3, len(scores))
    top_idx = np.argsort(-scores)[:search_k]
    res = []
    for idx in top_idx:
        score = float(scores[idx])
        if score >= SIMILARITY_THRESHOLD:
            meta = index["metadatas"][int(idx)].copy()
            meta["score"] = score
            res.append(meta)
    return res[:top_k]


def ingest_trusted_sites():
    print("🔄 Ingesting trusted medical websites...")
    success = 0
    for item in TRUSTED_SITES:
        url, category = item["url"], item["category"]
        try:
            title, text = scrape_url(url)
            chunks = chunk_text(text)
            if not chunks:
                continue
            embs = embed_texts(chunks)
            metas: List[dict] = []
            for i, chunk in enumerate(chunks):
                metas.append({
                    "source_title": title,
                    "url": url,
                    "chunk_id": f"{url}#chunk{i}",
                    "text": chunk,
                    "category": category,
                    "ingested_at": time.time(),
                })
            add_to_index(embs, metas)
            success += 1
            print(f"✅ Ingested {len(chunks)} chunks from {url}")
        except Exception as e:
            print(f"❌ Failed: {url} - {e}")

    if success > 0:
        save_index()
        print(f"✅ Ingested {len(index['metadatas'])} chunks from {success} sources")


# ---------- Endpoints ----------
@app.post("/ingest_direct")
def ingest_direct(req: DirectIngestRequest):
    chunks = chunk_text(req.content)
    if not chunks:
        raise HTTPException(400, "No content")
    embs = embed_texts(chunks)
    metas = [{
        "source_title": req.title,
        "url": req.source,
        "chunk_id": f"{req.source}#{req.title}#chunk{i}",
        "text": chunk,
        "category": req.category,
        "ingested_at": time.time(),
    } for i, chunk in enumerate(chunks)]
    add_to_index(embs, metas)
    save_index()
    return {"added_chunks": len(chunks), "title": req.title}


@app.post("/ingest_url")
def ingest_url(req: IngestRequest):
    try:
        title, text = scrape_url(req.url)
    except Exception as e:
        raise HTTPException(400, f"Failed: {e}")
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "No content")
    embs = embed_texts(chunks)
    metas = [{
        "source_title": title,
        "url": req.url,
        "chunk_id": f"{req.url}#chunk{i}",
        "text": chunk,
        "category": "User Ingested",
        "ingested_at": time.time()
    } for i, chunk in enumerate(chunks)]
    add_to_index(embs, metas)
    save_index()
    return {"added_chunks": len(chunks), "url": req.url, "title": title}


@app.post("/build_index")
def build_index_endpoint(req: BuildIndexRequest):
    return {"saved_to": save_index(), "num_docs": len(index["metadatas"])}


@app.get("/index_info")
def index_info():
    samples = []
    if index["metadatas"]:
        for meta in index["metadatas"][:5]:
            samples.append({
                "chunk_id": meta.get("chunk_id", ""),
                "source": meta.get("source_title", "")[:100],
                "category": meta.get("category", ""),
                "text_preview": meta.get("text", "")[:150],
            })
    return {
        "num_chunks": len(index["metadatas"]) if index["metadatas"] else 0,
        "dim": index["dim"],
        "has_faiss": _HAS_FAISS,
        "sample_content": samples
    }


@app.post("/triage_step")
def triage_step(req: TriageRequest):
    try:
        retrieved = retrieve(req.history, top_k=TOP_K)

        if retrieved:
            evidence_texts = []
            for r in retrieved:
                evidence_texts.append(
                    f"[Score: {r.get('score', 0):.2f}] {r.get('source_title', '')}\n"
                    f"{r.get('text', '')[:500]}"
                )
            evidence_block = "\n\n---\n\n".join(evidence_texts)
        else:
            evidence_block = "No medical evidence found."

        # CRITICAL FIX: Simpler, clearer prompt that demands proper JSON
        system_prompt = """You are a medical triage assistant. 

RESPOND ONLY WITH VALID JSON IN THIS EXACT FORMAT (no markdown, no code blocks):
{
  "candidates": [
    {"condition": "Condition Name 1", "probability": 0.5},
    {"condition": "Condition Name 2", "probability": 0.3},
    {"condition": "Condition Name 3", "probability": 0.2}
  ],
  "next_question": "Your follow-up question here",
  "top_recommendation": "Advice for most likely condition",
  "evidence_used": true,
  "evidence_reasoning": "Explain if/how you used the evidence"
}

RULES:
- ALWAYS include at least 2-3 candidates
- Each candidate MUST have "condition" (string) and "probability" (number 0-1)
- Probabilities should sum to ~1.0
- Use the medical evidence if relevant
- Return ONLY the JSON, nothing else"""

        user_prompt = f"""PATIENT: {req.history}

MEDICAL EVIDENCE:
{evidence_block}

Provide triage analysis in JSON format."""

        print(f"[DEBUG] Calling GPT-4...")
        response = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1000,
        )

        content = response.choices[0].message.content.strip()
        print(f"[DEBUG] Response ({len(content)} chars):\n{content[:300]}...")

        # Parse JSON with robust error handling
        parsed = None
        try:
            parsed = json.loads(content)
        except Exception as e1:
            print(f"[DEBUG] Parse error: {e1}")
            # Strip markdown fences
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            # Extract JSON object
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                json_str = content[start:end + 1]
                try:
                    parsed = json.loads(json_str)
                    print(f"[DEBUG] Successfully parsed after extraction")
                except Exception as e2:
                    print(f"[ERROR] Still can't parse: {e2}")
                    print(f"[ERROR] Attempted to parse: {json_str[:200]}...")
                    raise HTTPException(500, f"Invalid JSON from LLM")
            else:
                raise HTTPException(500, "No JSON found in response")

        if not parsed:
            raise HTTPException(500, "Failed to parse response")

        # CRITICAL: Validate candidates structure
        print(f"[DEBUG] Parsed keys: {list(parsed.keys())}")
        print(f"[DEBUG] Candidates: {parsed.get('candidates')}")

        candidates = parsed.get("candidates", [])
        if not isinstance(candidates, list):
            print(f"[ERROR] Candidates is not a list: {type(candidates)}")
            parsed["candidates"] = []
        else:
            valid = []
            for i, c in enumerate(candidates):
                print(f"[DEBUG] Candidate {i}: type={type(c)}, value={c}")
                if isinstance(c, dict) and "condition" in c and "probability" in c:
                    valid.append({
                        "condition": str(c["condition"]),
                        "probability": float(c["probability"])
                    })
                else:
                    print(f"[WARN] Invalid candidate: {c}")
            parsed["candidates"] = valid
            print(f"[DEBUG] Valid candidates: {len(valid)}")

        # Ensure required fields exist
        if "next_question" not in parsed:
            parsed["next_question"] = "DIAGNOSIS_COMPLETE"
        if "top_recommendation" not in parsed:
            parsed["top_recommendation"] = "Consult a healthcare professional."
        if "evidence_used" not in parsed:
            parsed["evidence_used"] = False

        parsed["_retrieved"] = retrieved
        parsed["_evidence_available"] = len(retrieved) > 0

        return parsed

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}")
        raise HTTPException(500, f"Error: {str(e)}")


# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    if not load_index():
        print("📦 Ingesting trusted medical sites...")
        ingest_trusted_sites()
    else:
        print(f"✅ Loaded RAG index: {len(index['metadatas'])} chunks")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)