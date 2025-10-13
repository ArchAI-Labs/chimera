# DALL-E TOOLS:
from crewai.tools import tool
import wget

@tool("Download Image Tool")
def download_image_tool(url: str, fname: str) -> str:
    """Tool to download an image given its url and save it using the passed fname."""
    ret = None
    try:
        ret = wget.download(url, fname)
    except Exception as e:
        print('Error during download:', e)
    if ret != fname:
        return f"There was an error during the download"
    return f"Image successfully saved as {fname}"