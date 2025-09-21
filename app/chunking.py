"""Recursive-ish text chunking for document ingestion.

Splits on paragraph/sentence boundaries first and only falls back to hard
character slicing when a single unit still exceeds chunk_size, which keeps
chunks semantically coherent -- the same strategy LangChain's
RecursiveCharacterTextSplitter uses. Implemented natively here so the
ingestion path has zero heavy dependencies; `requirements-optional.txt`
documents how to swap in `langchain-text-splitters` directly if desired.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    start_char: int


def _split_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]
    sentences = _SENT_SPLIT.split(paragraph)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sent) > chunk_size:
                for i in range(0, len(sent), chunk_size):
                    chunks.append(sent[i : i + chunk_size])
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split `text` into overlapping chunks suitable for embedding/retrieval."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    text = text.strip()
    if not text:
        return []

    paragraphs = [p for p in _PARA_SPLIT.split(text) if p.strip()]
    raw_pieces: list[str] = []
    for para in paragraphs:
        raw_pieces.extend(_split_paragraph(para.strip(), chunk_size))

    # Merge small adjacent pieces up to chunk_size, applying overlap between
    # consecutive merged chunks so retrieval doesn't lose context at seams.
    merged: list[str] = []
    current = ""
    for piece in raw_pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)

    chunks: list[Chunk] = []
    cursor = 0
    for i, piece in enumerate(merged):
        if i > 0 and chunk_overlap > 0:
            prev_tail = merged[i - 1][-chunk_overlap:]
            piece_with_overlap = f"{prev_tail} {piece}".strip()
        else:
            piece_with_overlap = piece
        chunks.append(Chunk(text=piece_with_overlap, index=i, start_char=cursor))
        cursor += len(piece)

    return chunks
