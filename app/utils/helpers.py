import os
from app.logger import get_logger

log = get_logger("utils")

def allowed_file(filename: str) -> bool:
    """Check if the file extension is supported (PDF, TXT, DOCX)."""
    allowed_extensions = {".pdf", ".txt", ".docx"}
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed_extensions

def validate_file_size(file_bytes_len: int, max_mb: int = 10) -> bool:
    """Verify that file size does not exceed the allowed threshold."""
    size_mb = file_bytes_len / (1024 * 1024)
    if size_mb > max_mb:
        log.warning(f"File size {size_mb:.2f}MB exceeds allowed threshold of {max_mb}MB.")
        return False
    return True

def format_file_size(bytes_len: int) -> str:
    """Format file sizes into a human-readable string (KB/MB)."""
    kb = bytes_len / 1024
    if kb > 1024:
        return f"{kb/1024:.2f} MB"
    return f"{kb:.2f} KB"
