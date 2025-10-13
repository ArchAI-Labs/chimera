from crewai.tools import tool
@tool("Text Chunker Tool")
def text_chunker_tool(text: str, chunk_size: int = 1000, overlap: int = 100):
    """Splits a long text into several pieces of fixed size."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks