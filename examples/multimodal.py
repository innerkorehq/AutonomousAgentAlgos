"""Multimodal: send an image with a question; the bytes live in the blob store, not in the log. Offline."""

import asyncio
import base64
import random
import struct
import zlib

from aaa.algos import ToolLoop
from aaa.data.blob import InMemoryBlobStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.llm.parts import ImagePart
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry


def png(width: int, height: int) -> bytes:
    """A real greyscale PNG (so the example needs no image file)."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    noise = random.Random(7)
    rows = b"".join(
        b"\x00" + noise.randbytes(width) for _ in range(height)
    )  # noise does not compress to nothing
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def model(request):  # stands in for a vision model
    images = [p for m in request.messages for p in m.parts or [] if p.type == "image"]
    return text(
        f"I can see {len(images)} image ({images[0].media_type}, larger than the inline limit: {images[0].size > 4096})."
    )


async def main() -> None:
    picture = png(160, 160)
    blobs = InMemoryBlobStore()
    registry = ToolRegistry([])
    algo = ToolLoop(registry)
    async with Engine(blobs=blobs) as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(registry).attach(engine)
        handle = await engine.start(
            algo,
            {
                "task": "What is in this picture?",
                "attachments": [ImagePart.from_bytes(picture, "image/png")],
            },
        )
        outcome = await handle.result()
        log = "".join(e.model_dump_json() for e in await engine.store.read(handle.run_id))
        print("answer:", outcome.value["text"])
        print("picture bytes in the log:", base64.b64encode(picture).decode() in log)
        print("blobs stored:", len(blobs._d))
        await engine.erase(handle.run_id)
        print("blobs after erasing the run:", len(blobs._d))


asyncio.run(main())
