from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_a_single_chunk():
    text = "This is a short document."
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_long_text_is_split_into_multiple_chunks():
    text = "\n\n".join(f"Paragraph {i}. " + "Filler sentence content here. " * 10 for i in range(10))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        # allow some slack for the overlap prefix appended to each chunk
        assert len(c.text) <= 200 + 40


def test_chunk_overlap_shares_content_between_consecutive_chunks():
    text = "\n\n".join(f"Paragraph {i} content filler words repeated many times over. " * 5 for i in range(6))
    chunks = chunk_text(text, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 2
    # the tail of chunk N should appear at the start of chunk N+1
    tail_of_first = chunks[0].text[-30:].strip()
    assert any(tail_of_first[:15] in chunks[i].text for i in range(1, len(chunks)))


def test_very_long_single_sentence_is_hard_split():
    text = "word " * 2000  # one giant "sentence" with no punctuation
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 + 20 for c in chunks)
