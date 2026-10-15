# Images, audio and files

A message can carry **attachments** (images, audio, files) after its text. They travel through the engine like any other part of the conversation, and leave it
only at the provider boundary. The threat model behind the choices below is on the [multimodal security page](../security/multimodal.md).

```python
--8<-- "examples/multimodal.py"
```

```text
answer: I can see 1 image (image/png, larger than the inline limit: True).
picture bytes in the log: False
blobs stored: 1
blobs after erasing the run: 0
```

## Sending attachments

```python
from aaa.llm.parts import ImagePart, AudioPart, FilePart

ImagePart.from_path("cat.png")                         # media type guessed from the name
ImagePart.from_bytes(raw, "image/jpeg", detail="low")
ImagePart.from_url("https://example.com/a.png", "image/png")   # the PROVIDER fetches it
FilePart.from_path("report.pdf")
AudioPart.from_bytes(wav, "audio/wav")

await engine.run(algo, {"task": "What is this?", "attachments": [ImagePart.from_path("cat.png")]})
await chat.send("And this one?", attachments=[ImagePart.from_path("dog.png")])      # sessions remember them
Message(role="user", content="Compare", parts=[...])                                  # or build messages yourself
```

Checks run when an attachment is created, so a bad one fails at your code, not inside a provider call: the declared type must fit the kind (`image/*` for
images, `audio/*` for audio), common image formats are checked against their leading bytes, only `https` URLs are accepted, and there is a size limit (20 MiB).
File names are display metadata only (the directory part is dropped; AAA never uses it as a path).

## Where the bytes live

- **With a blob store** (`Engine(blobs=FileBlobStore("./blobs"))`), attachments of 4 KiB or more are moved to it **before the event is appended**: the log holds a
  reference and a size, never base64. The provider-bound copy of the request gets the bytes back for that one call.
- **Without one**, small attachments (up to 256 KiB) stay inline in the event; larger ones are refused with an error that says to configure a blob store.
- Attachment blobs are **salted per run**, so the same image in two runs is two blobs: erasing one run (`engine.erase`) deletes exactly its own and cannot reveal that
  another run holds the same picture. Erasure removes them with the run.
- Traces record an attachment's kind, type and size only; never its bytes or blob reference.

## Providers and modalities

A client declares what it accepts (`client.modalities`); a request with an attachment the client cannot send fails with `UnsupportedModality` (a normal `aaa.llm.failed`
fact, so the run ends as `failed` with the reason) rather than silently dropping the picture.

| Client | Default modalities | Wire format |
|---|---|---|
| `OpenAICompatClient` (and LiteLLM) | text, image | chat-completions content parts: `image_url`, `input_audio` (wav, mp3), `file` |
| `OpenAIResponsesClient` | text, image, file | `input_image`, `input_file` |
| `FakeLLMClient` | everything | records the parts it was given |

Your gateway's model may accept more or less: set `LLMConfig(modalities=["image", "audio"])`. The wire shapes are checked against the OpenAI SDK's own types in CI;
nothing was sent to a live model.

## What does not see attachments

- **Guardrails read text.** An image can carry an injected instruction that no text guard sees. `GuardContext.has_media` tells a guard (or your policy) that the message
  carries attachments, and a message that is only an attachment is still passed to the guards, with empty text. Decide per policy: block, flag, or require approval.
- **The privacy redactor** (patterns, Presidio) acts on text; attachment bytes are left alone (a pattern could corrupt them) and cannot be redacted.
- **Tool results are text.** A tool cannot return an image to the model yet, and MCP image blocks are still rendered as `[image]`.

## Long conversations

Images and audio are expensive. The token counter charges a flat estimate per attachment (image 800, audio 1500, file 2000 tokens; override with
`HeuristicCounter(media=...)`), and the default context policy now includes `ClearMedia`, which replaces the attachments of older messages with a short note
(keeping the most recent one) before it touches tool results or summarises. The provider's reported usage remains the truth for spend.
