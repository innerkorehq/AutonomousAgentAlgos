# Threat model: multimodal messages

Written before the code (milestone M2). It says what we defend, what we do not, and which tests prove the first list.

## What changes

A message can carry **attachments** (images, audio, files) in addition to its text. Attachment bytes enter the system from the caller, travel through the event log, and
leave to a model provider. Each hop is a place to go wrong.

## Assets

- the user's media (often more sensitive than text: faces, documents, voices);
- the event log and blob store, which must stay small, durable and erasable;
- tenant isolation: one tenant's media must never be readable, deletable, or even detectable by another.

## Threats and decisions

| # | Threat | Decision | Test |
|---|---|---|---|
| 1 | **Bloat / denial of service**: a huge base64 string in a request fills the log, the bus and memory. | Every attachment has a size limit (default 20 MiB, configurable); bytes above a small inline threshold are moved to the blob store **before** the event is appended, so events carry a reference, never base64. With no blob store configured, only small attachments (default 256 KiB) are accepted and larger ones are refused with an error that names the fix. | oversize refused; log events stay small |
| 2 | **Content-type confusion**: a file labelled `image/png` that is something else. | The declared media type must match the attachment kind (`image/*` for images, `audio/*` for audio), and for common image formats the leading bytes are checked against the declared type. We never decode media, so decompression bombs are not our exposure. | mismatched signature refused |
| 3 | **URL attachments** make the *provider* fetch an address chosen by the user: an exfiltration channel (query string), a tracking pixel, or a probe of internal hosts if the provider runs in your network. | Only `https` URLs (and `http` when explicitly allowed) are accepted; `file:`, `data:`, `ftp:` and the like are refused. We do not fetch URLs ourselves. The docs say: a URL attachment is a request the provider will make on your behalf; prefer bytes for untrusted users. | scheme allow-list |
| 4 | **Cross-run and cross-tenant blob sharing**: content-addressed storage deduplicates, so two runs holding the same image share one blob; erasing one run would delete the other's, and a probe could learn that someone else holds that image. | Attachment blobs are **salted with their stream id** before hashing (the same technique as the existing claim-check), so identical media in two runs are two blobs; erasing a run deletes exactly its own. | erase one run, the other still reads its identical image |
| 5 | **Erasure**: media must disappear with the run. | The blob references are recorded in the event's `blobs`, which `engine.erase` already deletes (before scrubbing the events). Provider-held copies are outside our reach and are said so in the docs. | erase removes the blob |
| 6 | **Logs and traces leaking bytes**: capturing content for tracing would embed the image. | The trace derivation never includes attachment bytes or data: it records a placeholder with kind, media type and size. | trace of an image run has no base64 |
| 7 | **Guardrails and redaction do not see media**: an image can carry an injected instruction or personal data that the text guards and the PII redactor cannot read. | Not solved, stated: text guards, `Redactor` and Presidio act on text only. Mitigation offered: `GuardContext.has_media` lets a guard (or a policy) block or flag any message that carries media, and context management can drop old attachments. | guards receive `has_media` |
| 8 | **Unsupported modality silently dropped** (the model never saw the image, and answers anyway). | A client declares which modalities it supports; a request with an attachment it cannot take fails with `UnsupportedModality`, a normal `aaa.llm.failed` fact. Nothing is dropped silently. | failure is explicit |
| 9 | **Filename tricks** (`../../etc/passwd`) if a file attachment is later written to disk. | The filename is metadata only: stored as the base name, never used by AAA as a path. | base name only |
| 10 | **Cost**: images and audio consume large, provider-specific token counts that the heuristic counter cannot know. | Context budgeting charges a configurable flat estimate per attachment (images, audio, files) and offers a strategy that drops older attachments first. The provider's reported usage stays the truth for spend. | counter charges media |

## Out of scope

Malware scanning of uploaded files, OCR or transcription (media is passed through to capable providers), watermark/deepfake detection, and what a provider retains after it has
received the media (check its data-retention terms).
