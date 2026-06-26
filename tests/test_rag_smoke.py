from langchain_core.documents import Document
from app.rag.loaders import load_document_from_bytes
from app.rag.splitter import split_documents
from app.utils.helpers import allowed_file, validate_file_size

def test_allowed_file():
    """Verify that file extension validation works correctly."""
    assert allowed_file("itinerary.pdf") is True
    assert allowed_file("ticket.docx") is True
    assert allowed_file("notes.txt") is True
    assert allowed_file("unsupported.csv") is False
    assert allowed_file("malicious.exe") is False
    # Case insensitivity test
    assert allowed_file("ITINERARY.PDF") is True


def test_validate_file_size():
    """Verify that file size constraints are enforced."""
    # 1 MB should pass easily
    assert validate_file_size(1 * 1024 * 1024, max_mb=10) is True
    # 20 MB should fail a 10 MB limit
    assert validate_file_size(20 * 1024 * 1024, max_mb=10) is False


def test_txt_loader_smoke():
    """Smoke test text parsing from byte streams."""
    raw_content = b"This is a travel packing list containing sunglasses, sunscreen, and sandals."
    docs = load_document_from_bytes(raw_content, "packing.txt")
    
    assert len(docs) == 1
    assert docs[0].page_content == "This is a travel packing list containing sunglasses, sunscreen, and sandals."
    assert docs[0].metadata["source"] == "packing.txt"


def test_splitter_smoke():
    """Smoke test document chunking output size."""
    doc = Document(
        page_content="Travel info. " * 300,  # Generates ~3600 characters
        metadata={"source": "long_info.txt"}
    )
    chunks = split_documents([doc], chunk_size=1000, chunk_overlap=100)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 1100  # allowing for word boundaries
        assert chunk.metadata["source"] == "long_info.txt"
