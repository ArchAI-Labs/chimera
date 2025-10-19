from crewai.tools import tool
import wget
# ------------------------------------------------------------
# This module defines a custom CrewAI tool for downloading images
# from a given URL and saving them locally. It is mainly used by
# the Designer agent within the LinkedIn Crew to handle visual assets.
# ============================================================
@tool("Download Image Tool")
def download_image_tool(url: str, fname: str) -> str:
    """Tool to download an image given its url and save it using the passed fname."""
    ret = None
    try:
        ret = wget.download(url, fname)
    except Exception as e:
        print("Error during download:", e)
    if ret != fname:
        return f"There was an error during the download"
    return f"Image successfully saved as {fname}"
