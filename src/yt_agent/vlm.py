"""Optional VLM (Vision Language Model) frame description.

Pluggable backends — off by default. Enable via:
  CLI:    --vlm --vlm-backend ollama --vlm-model llava
  API:    ExtractOptions(vlm=True, vlm_backend="ollama", vlm_model="llava")

Supported backends:
  ollama      Local Ollama server (default: http://localhost:11434)
  openai      OpenAI-compatible API (OpenAI, llama.cpp server, LM Studio, etc.)

Each backend receives a JPEG frame as base64 and returns a text description.
Descriptions are written into FrameInfo.scene_label and persisted to
<output_dir>/frames/<stem>.txt alongside each image.
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path

from yt_agent.schemas import FrameInfo


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class VLMBackend(ABC):
    """Abstract vision-language model backend."""

    @abstractmethod
    def describe(self, image_path: Path, prompt: str) -> str:
        """Return a text description of the image at image_path."""


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaBackend(VLMBackend):
    """Describe frames via a local Ollama server.

    Requires a multimodal model installed, e.g.:
        ollama pull llava
    """

    def __init__(
        self,
        model: str = "llava",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def describe(self, image_path: Path, prompt: str) -> str:
        import urllib.request

        image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "").strip()
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# OpenAI-compatible backend  (OpenAI, llama.cpp server, LM Studio, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleBackend(VLMBackend):
    """Describe frames via an OpenAI-compatible vision API.

    Works with:
      - OpenAI (api_base=https://api.openai.com/v1, model=gpt-4o)
      - llama.cpp server (api_base=http://localhost:8080/v1)
      - LM Studio (api_base=http://localhost:1234/v1)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def describe(self, image_path: Path, prompt: str) -> str:
        import urllib.request

        image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 256,
        }).encode()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise RuntimeError(f"OpenAI-compatible API request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_backend(
    backend: str = "ollama",
    model: str | None = None,
    api_base: str | None = None,
    api_key: str = "",
) -> VLMBackend:
    """Construct a VLMBackend from string config.

    Args:
        backend: "ollama" or "openai".
        model: Model name override. Defaults: ollama→"llava", openai→"gpt-4o".
        api_base: API base URL override.
        api_key: API key (openai backend only).
    """
    if backend == "ollama":
        return OllamaBackend(
            model=model or "llava",
            base_url=api_base or "http://localhost:11434",
        )
    elif backend == "openai":
        return OpenAICompatibleBackend(
            model=model or "gpt-4o",
            api_base=api_base or "https://api.openai.com/v1",
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown VLM backend {backend!r}. Use: ollama, openai")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = (
    "Describe what is visible in this video frame in one or two sentences. "
    "Focus on the main subject, action, and setting."
)


def describe_frames(
    frames: list[FrameInfo],
    output_dir: Path,
    backend: VLMBackend,
    prompt: str = DEFAULT_PROMPT,
) -> list[FrameInfo]:
    """Run VLM inference on each frame and return updated FrameInfo list.

    Descriptions are stored in FrameInfo.scene_label and saved as
    <frame_stem>.txt next to each image.

    Args:
        frames: List of FrameInfo from extract_frames().
        output_dir: Root output dir (frames are at output_dir/frames/).
        backend: VLMBackend instance to use.
        prompt: Instruction prompt sent to the VLM for each frame.

    Returns:
        New list of FrameInfo with scene_label populated.
    """
    updated: list[FrameInfo] = []
    for fi in frames:
        img_path = output_dir / fi.path
        if not img_path.exists():
            updated.append(fi)
            continue

        try:
            description = backend.describe(img_path, prompt)
        except RuntimeError:
            description = ""

        # Persist alongside image
        txt_path = img_path.with_suffix(".txt")
        if description:
            txt_path.write_text(description, encoding="utf-8")

        updated.append(fi.model_copy(update={"scene_label": description or None}))

    return updated
