"""
DALL-E image generation tool for LlamaIndex
Migrated from CrewAI implementation
"""
from typing import Optional
import os
import requests
from llama_index.core.tools import FunctionTool


class DallETool:
    """
    DALL-E image generation tool compatible with LlamaIndex.
    If no API key is configured, calls to `run` will be skipped gracefully.
    """

    def __init__(
        self,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        api_key: Optional[str] = None
    ):
        """
        Initialize DALL-E tool

        Args:
            model: DALL-E model version
            size: Image size
            quality: Image quality
            n: Number of images to generate
            api_key: OpenAI API key (defaults to env var)
        """
        self.model = model
        self.size = size
        self.quality = quality
        self.n = n
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Instead of raising, become a no-op tool when missing the key
        self.enabled = bool(self.api_key)
        self._disabled_reason = (
            "OPENAI_API_KEY not set; skipping image generation."
            if not self.enabled else ""
        )

    def run(self, prompt: str, output_dir: str = "output/images") -> str:
        """
        Generate an image using DALL-E

        Args:
            prompt: Text description of the image to generate
            output_dir: Directory to save generated images

        Returns:
            Path/URL info string on success, or a skip/explanatory message.
        """
        # If disabled, quietly skip and explain
        if not self.enabled:
            return f"Skipped image generation: {self._disabled_reason}"

        try:
            import openai

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Generate image
            client = openai.OpenAI(api_key=self.api_key)
            response = client.images.generate(
                model=self.model,
                prompt=prompt,
                size=self.size,
                quality=self.quality,
                n=self.n
            )

            # Prefer URL when available; some SDKs may return b64 instead.
            image_url = getattr(response.data[0], "url", None)
            b64 = getattr(response.data[0], "b64_json", None)

            image_filename = f"{output_dir}/dalle_{hash((prompt, self.size, self.quality, self.n)) % 100000}.png"

            if image_url:
                img_response = requests.get(image_url, timeout=30)
                img_response.raise_for_status()
                with open(image_filename, "wb") as f:
                    f.write(img_response.content)
                return f"Image generated successfully: {image_filename}\nURL: {image_url}"

            elif b64:
                import base64
                with open(image_filename, "wb") as f:
                    f.write(base64.b64decode(b64))
                return f"Image generated successfully: {image_filename}"

            else:
                return "Error generating image: No URL or base64 image returned by API."

        except Exception as e:
            return f"Error generating image: {str(e)}"

    def as_tool(self) -> FunctionTool:
        """
        Convert to LlamaIndex FunctionTool.
        If disabled, the tool still registers but returns a skip message when called.
        """
        desc = (
            "Generate an image using the DALL-E AI model. "
            "Input should be a detailed text description of the image you want to create. "
            "Returns the path to the saved image file."
        )
        if not self.enabled:
            desc += " (Currently disabled: missing OPENAI_API_KEY; calls will be skipped.)"

        return FunctionTool.from_defaults(
            fn=self.run,
            name="generate_image_dalle",
            description=desc
        )


def download_image_tool(url: str, output_dir: str = "output/images") -> str:
    """
    Download an image from a URL

    Args:
        url: URL of the image to download
        output_dir: Directory to save the image

    Returns:
        Path to the downloaded image or error message
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Download image
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Determine filename from URL
        filename = url.split("/")[-1].split("?")[0]
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            filename = f"image_{hash(url) % 10000}.png"

        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(response.content)

        return f"Image downloaded successfully: {filepath}"

    except Exception as e:
        return f"Error downloading image: {str(e)}"
