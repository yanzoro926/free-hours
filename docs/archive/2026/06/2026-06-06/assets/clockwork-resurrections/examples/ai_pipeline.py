"""
Example 2: AI Pipeline — chunk text, generate embeddings, search.

Demonstrates a realistic AI workflow:
1. Ingest: load documents and split into chunks
2. Embed: generate vector embeddings per chunk (simulated)
3. Index: organize chunks with metadata
4. Search: query the indexed data (simulated)
"""

import sys
import os
import time
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from durable_mini import DurableEngine, Workflow

DB_PATH = "/tmp/durable_ai_pipeline.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

engine = DurableEngine(DB_PATH)

# Sample documents
DOCUMENTS = [
    "Durable execution ensures workflows survive crashes by checkpointing state to a database after each step.",
    "pg_durable is Microsoft's new PostgreSQL extension that brings durable execution inside the database.",
    "Temporal.io is a popular open-source durable execution platform used by Netflix, Snap, and Stripe.",
    "Azure Durable Functions provides serverless durable execution on the Azure cloud platform.",
    "The key insight: if you save inputs and outputs to a database, you can always replay from the last checkpoint.",
    "Vector embeddings transform text into numerical representations that capture semantic meaning.",
    "pgvector is a PostgreSQL extension that enables efficient vector similarity search.",
    "RAG (Retrieval-Augmented Generation) combines vector search with LLMs for grounded responses.",
]


@engine.step(description="Load and chunk documents")
def ingest_documents() -> list[dict]:
    """Split documents into chunks with metadata."""
    chunks = []
    for i, doc in enumerate(DOCUMENTS):
        # Simple chunking: each sentence is a chunk
        chunk = {
            "id": f"chunk_{i:03d}",
            "doc_id": i,
            "text": doc,
            "char_count": len(doc),
        }
        chunks.append(chunk)
        time.sleep(0.01)  # Simulate I/O
    return chunks


@engine.step(description="Generate embedding for each chunk")
def embed_chunk(chunk: dict) -> dict:
    """Simulate embedding generation (in reality: call an embedding API)."""
    time.sleep(0.02)  # Simulate API call
    # Deterministic "embedding" from text hash
    text_hash = hashlib.sha256(chunk["text"].encode()).hexdigest()
    # Simulate a 4-dimensional embedding
    embedding = [
        int(text_hash[i : i + 4], 16) / 65535.0 for i in range(0, 16, 4)
    ]
    return {**chunk, "embedding": embedding}


@engine.step(description="Index chunks for retrieval")
def build_index(chunks: list[dict]) -> dict:
    """Build a searchable index from embedded chunks."""
    index = {}
    for chunk in chunks:
        index[chunk["id"]] = {
            "text": chunk["text"],
            "embedding": chunk["embedding"],
            "char_count": chunk["char_count"],
        }
    return {
        "index": index,
        "total_chunks": len(chunks),
        "total_chars": sum(c["char_count"] for c in chunks),
    }


@engine.step(description="Simulate semantic search against index")
def search_index(index_data: dict) -> list[dict]:
    """Simulate a search query (in reality: cosine similarity)."""
    query_terms = ["durable", "execution"]
    results = []
    for chunk_id, chunk in index_data["index"].items():
        text_lower = chunk["text"].lower()
        score = sum(1 for term in query_terms if term in text_lower)
        if score > 0:
            results.append(
                {
                    "id": chunk_id,
                    "text": chunk["text"][:80] + "...",
                    "score": score,
                }
            )
    # Sort by relevance
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# === Build workflow ===

workflow = Workflow(
    "ai_embedding_pipeline",
    description="Ingest → Embed → Index → Search — durable AI pipeline",
)

workflow.add_step(ingest_documents)
workflow.add_step(embed_chunk, depends_on=["ingest_documents"], fan_out=True)
workflow.add_step(build_index, depends_on=["embed_chunk"])
workflow.add_step(search_index, depends_on=["build_index"])


def run_demo():
    print("=" * 60)
    print("  Durable Mini — AI Embedding Pipeline")
    print("=" * 60)

    result = engine.run(workflow, instance_id="ai-pipeline-001")
    print(f"\nPipeline: {result.status}")
    print(f"Duration: {result.duration_seconds:.3f}s")

    if result.status == "completed":
        index_data = result.output["build_index"]
        print(f"\n📊 Index Stats:")
        print(f"  Total chunks: {index_data['total_chunks']}")
        print(f"  Total characters: {index_data['total_chars']}")

        search_results = result.output["search_index"]
        print(f"\n🔍 Search Results for 'durable execution':")
        for r in search_results:
            print(f"  [{r['id']}] score={r['score']} — {r['text']}")

    print("\n[Step Status]")
    for step in engine.status("ai-pipeline-001")["steps"]:
        icon = "✅" if step["status"] == "completed" else "❌"
        print(f"  {icon} {step['name']}: {step['status']}")

    engine.close()
    os.remove(DB_PATH)


if __name__ == "__main__":
    run_demo()
