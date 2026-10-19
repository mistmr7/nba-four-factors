"""Retrieval and grounded answering.

RAG step three: embed the question with the same model as the chunks,
take the top-k nearest chunks by cosine similarity, and hand them to
the LLM with instructions to answer ONLY from that context and to
cite sources. Grounding plus citations is the whole point: the model
answers from your documents, and you can check it.

If ANTHROPIC_API_KEY is set, the answer comes from the Claude API.
If not, the assembled prompt is printed so it can be pasted into any
Claude interface; retrieval is the part that runs locally either way.
"""

from __future__ import annotations

import os

import numpy as np

from .index import load_index

MODEL = os.environ.get("RAG_MODEL", "claude-sonnet-5-5")
TOP_K = int(os.environ.get("RAG_TOP_K", "6"))

SYSTEM = (
    "You answer questions about Mike Mistarz's NBA analytics research "
    "using ONLY the provided context chunks. Cite the source file of "
    "every claim in [brackets]. If the context does not contain the "
    "answer, say so plainly rather than guessing."
)


def retrieve(question: str, k: int = TOP_K):
    chunks, vectors, emb = load_index()
    q = emb.encode([question])[0]
    sims = vectors @ q
    top = np.argsort(-sims)[:k]
    rows = chunks.iloc[top].copy()
    rows["similarity"] = sims[top]
    return rows


def build_prompt(question: str, rows) -> str:
    blocks = []
    for r in rows.itertuples(index=False):
        blocks.append(f"[{r.source} :: {r.section}]\n{r.text}")
    context = "\n\n---\n\n".join(blocks)
    return f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"


def answer(question: str, k: int = TOP_K) -> None:
    rows = retrieve(question, k)
    print("retrieved:")
    for r in rows.itertuples(index=False):
        print(f"  {r.similarity:.3f}  {r.source} :: {r.section}")
    prompt = build_prompt(question, rows)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nNo ANTHROPIC_API_KEY set. Paste this prompt into Claude:\n")
        print(SYSTEM + "\n\n" + prompt)
        return

    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    print("\n" + text)
