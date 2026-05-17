# lfm25-audio-onnx — LiquidAI LFM2.5 Audio (ONNX)

Standalone Path 3 Rust cdylib that registers `LFM25AudioOnnxNode` into the
[RemoteMedia SDK](https://github.com/RemoteMedia-SDK/remotemedia-sdk) streaming
pipeline registry.

This plugin owns the multi-graph ONNX generation loop (decoder + depthformer +
detokenizer, with optional GPU ISTFT) for the LFM2.5 Audio family —
extracted out of `remotemedia-core` so the host crate no longer drags in
`ort` for this single node. The streaming surface is unchanged: audio frames
plus interleaved text fragments cross the dlopen boundary through the
multi-output FFI path.

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["lfm25-audio-onnx@v0.1.0"],
  "nodes": [
    {
      "id": "lfm25",
      "node_type": "LFM25AudioOnnxNode",
      "params": {
        "model_dir": "models/LFM2.5-Audio-1.5B-ONNX",
        "mode": "interleaved",
        "precision": "q4",
        "device": "cuda:0",
        "audio_batch_size": 12,
        "first_chunk_audio_batch_size": 4
      }
    }
  ]
}
```

The SDK resolver expands `lfm25-audio-onnx@v0.1.0` to
`github.com/RemoteMedia-SDK/lfm25-audio-onnx`, fetches `plugin.toml`, then
falls through to `release-manifest.json` for the platform-specific prebuilt
`.so` / `.dylib` / `.dll` asset.

## Build the cdylib locally

```bash
git clone https://github.com/RemoteMedia-SDK/lfm25-audio-onnx
cd lfm25-audio-onnx
cargo build --release
# → target/release/liblfm25_audio_onnx_plugin.so
```

## What it exports

| Node type            | Input            | Output                                |
|----------------------|------------------|---------------------------------------|
| `LFM25AudioOnnxNode` | `Audio` / `Text` | `Audio` (24 kHz mono) + `Text` frames |

Aux-port control envelopes (`context`, `system_prompt`, `reset`) are
accepted via the standard `__aux_port__` JSON envelope and mutate the
node's per-session state.

Mode matrix:
- `asr` — audio in, text out
- `tts` — text in, audio out
- `interleaved` — text or audio in, interleaved text + audio out

## License

See `LICENSE.md`. Governed by the RemoteMedia SDK Community License 1.0.
