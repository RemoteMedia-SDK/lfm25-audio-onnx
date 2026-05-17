//! LiquidAI LFM2.5 Audio ONNX streaming node as a standalone Path 3
//! loadable plugin.
//!
//! This crate owns the multi-graph ONNX generation loop (decoder +
//! depthformer + detokenizer, with optional GPU ISTFT) for the LFM2.5
//! Audio family. Originally lived in `remotemedia-core` under
//! `nodes/onnx/lfm25_audio.rs`; extracted here so the host crate no
//! longer drags in `ort` for this single node.
//!
//! ## Node types exported
//!
//!   LFM25AudioOnnxNode — Audio | Text → interleaved Text / Audio
//!                        (24 kHz mono) with terminal `<|text_end|>` /
//!                        `<|audio_end|>` markers.
//!
//! Aux-port control envelopes (`context`, `system_prompt`, `reset`,
//! `barge_in`) are routed via the standard `"__aux_port__"` JSON
//! envelope and mutate the node's per-session state.
//!
//! ## Hand-rolled (not `#[node]`-generated)
//!
//! The original in-tree node used the `#[node]` proc-macro to flatten a
//! 17-field config onto the struct and auto-generate the
//! `AsyncStreamingNode` impl. Across the dlopen boundary that machinery
//! adds no value (the host never sees the schema, capabilities, or the
//! macro-generated factory), so this plugin writes the config struct,
//! the trait impl, and the `FfiNodeFactory` by hand.

use async_trait::async_trait;
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;
use remotemedia_plugin_sdk::types::{AudioSamples, Error, RuntimeData};
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};

use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, ExecutionProviderDispatch,
};
use ort::memory::Allocator;
use ort::session::{
    builder::GraphOptimizationLevel, OutputSelector, RunOptions, Session, SessionInputValue,
    SessionOutputs,
};
use ort::value::{DynTensorValueType, DynValue, Tensor};

// Aux-port envelope key (mirrors `crate::transport::session_control::AUX_PORT_ENVELOPE_KEY`
// in the host workspace — inlined here so the plugin doesn't depend on
// `remotemedia-core`).
const AUX_PORT_ENVELOPE_KEY: &str = "__aux_port__";

const TEXT_END: &str = "<|text_end|>";
const AUDIO_END: &str = "<|audio_end|>";
const DEFAULT_SYSTEM_PROMPT_ASR: &str = "Perform ASR.";
const DEFAULT_SYSTEM_PROMPT_TTS: &str = "Perform TTS. Use the UK female voice.";
const DEFAULT_SYSTEM_PROMPT_INTERLEAVED: &str = "Respond with interleaved text and audio.";

type L25Result<T> = ::core::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Mode / Precision enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LFM25AudioMode {
    Asr,
    Tts,
    Interleaved,
}

impl Default for LFM25AudioMode {
    fn default() -> Self {
        Self::Interleaved
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LFM25AudioPrecision {
    Q4,
    Fp16,
    Fp32,
}

impl Default for LFM25AudioPrecision {
    fn default() -> Self {
        Self::Q4
    }
}

impl LFM25AudioPrecision {
    fn suffix(self) -> &'static str {
        match self {
            Self::Q4 => "_q4",
            Self::Fp16 => "_fp16",
            Self::Fp32 => "",
        }
    }
}

// ---------------------------------------------------------------------------
// Config (hand-rolled — was previously macro-generated from `#[config(...)]`
// annotations on the node struct)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LFM25AudioOnnxConfig {
    /// Directory containing the unpacked model bundle. The node resolves
    /// ONNX graphs from `<modelDir>/onnx` when that directory exists,
    /// which matches the Liquid exporter layout.
    pub model_dir: PathBuf,

    /// Default mode for the node.
    pub mode: LFM25AudioMode,

    /// Selected ONNX precision family.
    pub precision: LFM25AudioPrecision,

    /// Persona / behavior prompt.
    pub system_prompt: String,

    /// ONNX Runtime execution device. Use "cuda:0", "cuda:1" for GPU
    /// acceleration, or "cpu" for CPU-only.
    pub device: String,

    /// Maximum number of decode steps delegated to the backend.
    pub max_new_tokens: usize,

    /// Expected ASR input sample rate.
    pub input_sample_rate: u32,

    /// Generated speech sample rate.
    pub output_sample_rate: u32,

    /// Steady-state audio frame grouping hint.
    pub audio_batch_size: usize,

    /// Optional smaller threshold for the first generated audio chunk.
    pub first_chunk_audio_batch_size: Option<usize>,

    /// Text-token sampling temperature.
    pub text_temperature: f32,

    /// Audio-code sampling temperature.
    pub audio_temperature: f32,

    /// Optional top-k cutoff for audio codebook sampling.
    pub audio_top_k: usize,

    /// When enabled in interleaved audio-input mode, force generation
    /// into the audio branch after the first decoder decision instead
    /// of waiting for the model to emit its audio-start marker naturally.
    pub audio_first_interleaved: bool,

    /// When enabled, load all ONNX sessions during `initialize()` and
    /// run a short synthetic warmup so first live inference avoids cold
    /// graph construction and kernel setup.
    pub prewarm: bool,

    /// Optional override for the depthformer (vocoder) precision.
    pub depthformer_precision: Option<LFM25AudioPrecision>,

    /// Optional override device for the depthformer (vocoder).
    pub depthformer_device: Option<String>,

    /// When false, skip the `_unrolled` depthformer variant even if it
    /// exists on disk.
    pub use_unrolled_depthformer: bool,
}

impl Default for LFM25AudioOnnxConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models/LFM2.5-Audio-1.5B-ONNX"),
            mode: LFM25AudioMode::Interleaved,
            precision: LFM25AudioPrecision::Q4,
            system_prompt: String::new(),
            device: String::from("cpu"),
            max_new_tokens: 4096,
            input_sample_rate: 16_000,
            output_sample_rate: 24_000,
            audio_batch_size: 12,
            first_chunk_audio_batch_size: None,
            text_temperature: 1.0,
            audio_temperature: 1.0,
            audio_top_k: 4,
            audio_first_interleaved: false,
            prewarm: false,
            depthformer_precision: None,
            depthformer_device: None,
            use_unrolled_depthformer: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct SessionState {
    context: String,
    system_prompt: String,
    turn_count: u64,
}

#[derive(Debug, Clone)]
enum LFM25Input {
    Audio {
        samples: Vec<f32>,
        sample_rate: u32,
        mode: LFM25AudioMode,
    },
    Text {
        text: String,
        mode: LFM25AudioMode,
    },
}

trait LFM25AudioBackend: Send + Sync {
    fn validate_bundle(&self, config: &LFM25AudioOnnxConfig) -> L25Result<()>;
    fn prewarm(&self, _config: &LFM25AudioOnnxConfig) -> L25Result<()> {
        Ok(())
    }
    /// Generate output, emitting each chunk via the callback as it
    /// becomes available. Returns the total number of items emitted.
    fn generate_streaming(
        &self,
        input: LFM25Input,
        config: &LFM25AudioOnnxConfig,
        emit: &mut dyn FnMut(RuntimeData) -> L25Result<()>,
    ) -> L25Result<usize>;
}

#[derive(Debug, Deserialize)]
struct EmbeddingMeta {
    vocab_size: usize,
    hidden_size: usize,
}

#[derive(Debug, Deserialize)]
struct ModelConfigFile {
    #[serde(default)]
    lfm: LfmCoreConfig,
}

#[derive(Debug, Deserialize)]
struct LfmCoreConfig {
    #[serde(default = "default_hidden_size")]
    hidden_size: usize,
    #[serde(default = "default_num_layers")]
    num_hidden_layers: usize,
    #[serde(default = "default_num_kv_heads")]
    num_key_value_heads: usize,
    #[serde(default = "default_num_heads")]
    num_attention_heads: usize,
    #[serde(default = "default_conv_cache")]
    conv_L_cache: usize,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
}

fn default_hidden_size() -> usize {
    2048
}
fn default_num_layers() -> usize {
    16
}
fn default_num_kv_heads() -> usize {
    8
}
fn default_num_heads() -> usize {
    32
}
fn default_conv_cache() -> usize {
    3
}
fn default_vocab_size() -> usize {
    65_536
}

fn effective_system_prompt(mode: LFM25AudioMode, config: &LFM25AudioOnnxConfig) -> &str {
    if !config.system_prompt.trim().is_empty() {
        return config.system_prompt.as_str();
    }

    match mode {
        LFM25AudioMode::Asr => DEFAULT_SYSTEM_PROMPT_ASR,
        LFM25AudioMode::Tts => DEFAULT_SYSTEM_PROMPT_TTS,
        LFM25AudioMode::Interleaved => DEFAULT_SYSTEM_PROMPT_INTERLEAVED,
    }
}

impl Default for LfmCoreConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_layers(),
            num_key_value_heads: default_num_kv_heads(),
            num_attention_heads: default_num_heads(),
            conv_L_cache: default_conv_cache(),
            layer_types: Vec::new(),
            vocab_size: default_vocab_size(),
        }
    }
}

// ---------------------------------------------------------------------------
// ONNX runtime (multi-graph)
// ---------------------------------------------------------------------------

struct OrtLFM25Runtime {
    tokenizer: Tokenizer,
    decoder: Mutex<Session>,
    decoder_token_output: bool,
    audio_embedding_session: Mutex<Session>,
    audio_detokenizer: Mutex<Session>,
    audio_istft: Option<Mutex<Session>>,
    depthformer: Mutex<Session>,
    depthformer_unrolled: bool,
    depthformer_token_output: bool,
    audio_encoder: Option<Mutex<Session>>,
    embed_tokens: Vec<f32>,
    embed_meta: EmbeddingMeta,
    audio_embedding: Option<(Vec<f32>, EmbeddingMeta)>,
    lfm: LfmCoreConfig,
    mel: MelConfig,
    mel_filters: Vec<Vec<(usize, f32)>>,
    mel_hann: Vec<f32>,
    mel_fft: Arc<dyn Fft<f32>>,
    detokenizer_window: Vec<f32>,
    detokenizer_ifft: Arc<dyn Fft<f32>>,
    prefix_caches: Mutex<HashMap<String, DecoderCache>>,
    text_temperature: f32,
    audio_temperature: f32,
    audio_top_k: usize,
    rng: Mutex<StdRng>,
}

struct DecoderCache {
    values: HashMap<String, DynValue>,
    total_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderReadout {
    Logits,
    HiddenStates,
}

struct DecoderStepOutput {
    token_id: Option<u32>,
    hidden_states: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct MelConfig {
    sample_rate: u32,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
    preemph: f32,
    log_zero_guard: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 128,
            fmin: 0.0,
            fmax: 8_000.0,
            preemph: 0.97,
            log_zero_guard: 5.960_464_5e-8,
        }
    }
}

impl OrtLFM25Runtime {
    const IM_END_TOKEN: u32 = 7;
    const AUDIO_START_TOKEN: u32 = 128;
    const TEXT_END_TOKEN: u32 = 130;
    const END_OF_AUDIO_TOKEN: i64 = 2048;
    const NUM_CODEBOOKS: usize = 8;
    const CODEBOOK_VOCAB: usize = 2049;
    const INTERLEAVED_AUDIO_RUN: usize = 12;
    const INTERLEAVED_TEXT_RUN: usize = 6;

    fn load(config: &LFM25AudioOnnxConfig) -> L25Result<Self> {
        let onnx_dir = OrtLFM25AudioBackend::onnx_dir(config);
        let model_root = if onnx_dir == config.model_dir {
            config
                .model_dir
                .parent()
                .unwrap_or(config.model_dir.as_path())
                .to_path_buf()
        } else {
            config.model_dir.clone()
        };

        let tokenizer_path = model_root.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            Error::Execution(format!(
                "LFM25AudioOnnxNode: failed to load tokenizer {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        let config_path = model_root.join("config.json");
        let lfm = if config_path.exists() {
            let text = fs::read_to_string(&config_path)?;
            serde_json::from_str::<ModelConfigFile>(&text)
                .map_err(|e| {
                    Error::Execution(format!(
                        "LFM25AudioOnnxNode: invalid {}: {e}",
                        config_path.display()
                    ))
                })?
                .lfm
        } else {
            LfmCoreConfig::default()
        };

        let embed_meta = read_meta(&onnx_dir.join("embed_tokens.json"))?;
        let embed_tokens = read_f32_blob(&onnx_dir.join("embed_tokens.bin"))?;
        let audio_embedding = {
            let bin = onnx_dir.join("audio_embedding.bin");
            let meta = onnx_dir.join("audio_embedding.json");
            if bin.exists() && meta.exists() {
                Some((read_f32_blob(&bin)?, read_meta(&meta)?))
            } else {
                None
            }
        };
        let mel = {
            let path = onnx_dir.join("mel_config.json");
            if path.exists() {
                serde_json::from_str::<MelConfig>(&fs::read_to_string(path)?).unwrap_or_default()
            } else {
                MelConfig::default()
            }
        };
        let mel_filters = mel_filterbank(&mel);
        let mel_hann = hann_window(mel.win_length);
        let mut planner = FftPlanner::<f32>::new();
        let mel_fft = planner.plan_fft_forward(mel.n_fft);
        let detokenizer_window = hann_window(1280);
        let detokenizer_ifft = planner.plan_fft_inverse(1280);

        let suffix = config.precision.suffix();
        let device = config.device.as_str();
        let istft_path = onnx_dir.join("audio_istft.onnx");
        let decoder_token_path = onnx_dir.join(format!("decoder{suffix}_argmax.onnx"));
        let force_text_sampling = config.text_temperature > 0.0
            && std::env::var_os("LFM25_DISABLE_TEXT_SAMPLING").is_none();
        let decoder_token_output = decoder_token_path.exists()
            && !force_text_sampling
            && std::env::var_os("LFM25_DISABLE_DECODER_ARGMAX").is_none();
        let decoder_path = if decoder_token_output {
            decoder_token_path
        } else {
            onnx_dir.join(format!("decoder{suffix}.onnx"))
        };
        let df_suffix = config
            .depthformer_precision
            .unwrap_or(config.precision)
            .suffix();
        let depthformer_unrolled_path =
            onnx_dir.join(format!("vocoder_depthformer{df_suffix}_unrolled.onnx"));
        let depthformer_token_path =
            onnx_dir.join(format!("vocoder_depthformer{df_suffix}_argmax.onnx"));
        let force_audio_sampling = config.audio_temperature > 0.0
            && std::env::var_os("LFM25_DISABLE_AUDIO_SAMPLING").is_none();
        let depthformer_unrolled_sample_path =
            onnx_dir.join(format!("vocoder_depthformer{df_suffix}_unrolled_sample.onnx"));
        let use_unrolled_sample = force_audio_sampling
            && config.use_unrolled_depthformer
            && depthformer_unrolled_sample_path.exists();
        let depthformer_unrolled = (!force_audio_sampling
            && config.use_unrolled_depthformer
            && depthformer_unrolled_path.exists())
            || use_unrolled_sample;
        let depthformer_token_output = !use_unrolled_sample
            && !depthformer_unrolled
            && !force_audio_sampling
            && depthformer_token_path.exists();
        let depthformer_path = if use_unrolled_sample {
            depthformer_unrolled_sample_path
        } else if depthformer_unrolled {
            depthformer_unrolled_path
        } else if depthformer_token_output {
            depthformer_token_path
        } else {
            onnx_dir.join(format!("vocoder_depthformer{df_suffix}.onnx"))
        };

        let df_device = config.depthformer_device.as_deref().unwrap_or(device);

        tracing::info!(
            device,
            precision = ?config.precision,
            decoder_argmax = decoder_token_output,
            depthformer_unrolled,
            depthformer_argmax = depthformer_token_output,
            depthformer_precision = df_suffix,
            depthformer_device = df_device,
            has_istft = istft_path.exists(),
            has_audio_embedding_bin = audio_embedding.is_some(),
            "LFM25AudioOnnxNode: loading ONNX sessions"
        );
        Ok(Self {
            tokenizer,
            decoder: Mutex::new(load_session(decoder_path, device)?),
            decoder_token_output,
            audio_embedding_session: Mutex::new(load_session(
                onnx_dir.join(format!("audio_embedding{suffix}.onnx")),
                device,
            )?),
            audio_encoder: Some(Mutex::new(load_session(
                onnx_dir.join(format!("audio_encoder{suffix}.onnx")),
                device,
            )?)),
            audio_detokenizer: Mutex::new(load_session(
                onnx_dir.join(format!("audio_detokenizer{suffix}.onnx")),
                device,
            )?),
            audio_istft: if istft_path.exists() {
                Some(Mutex::new(load_session(istft_path, device)?))
            } else {
                None
            },
            depthformer: Mutex::new(load_session(depthformer_path, df_device)?),
            depthformer_unrolled,
            depthformer_token_output,
            embed_tokens,
            embed_meta,
            audio_embedding,
            lfm,
            mel,
            mel_filters,
            mel_hann,
            mel_fft,
            detokenizer_window,
            detokenizer_ifft,
            prefix_caches: Mutex::new(HashMap::new()),
            text_temperature: config.text_temperature,
            audio_temperature: config.audio_temperature,
            audio_top_k: config.audio_top_k,
            rng: Mutex::new(StdRng::from_entropy()),
        })
    }

    fn encode(&self, text: &str) -> L25Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: tokenization failed: {e}")))
    }

    fn decode(&self, ids: &[u32]) -> L25Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: decode failed: {e}")))
    }

    fn text_embeddings(&self, ids: &[u32]) -> L25Result<Vec<f32>> {
        let hidden = self.embed_meta.hidden_size;
        let mut out = Vec::with_capacity(ids.len() * hidden);
        for &id in ids {
            let idx = id as usize;
            if idx >= self.embed_meta.vocab_size {
                return Err(Error::Execution(format!(
                    "LFM25AudioOnnxNode: token id {idx} outside embedding vocab"
                )));
            }
            let start = idx * hidden;
            out.extend_from_slice(&self.embed_tokens[start..start + hidden]);
        }
        Ok(out)
    }

    fn audio_embeddings_sum(&self, frame: &[i64; 8]) -> L25Result<Vec<f32>> {
        let hidden = self.lfm.hidden_size;
        if let Some((weights, meta)) = &self.audio_embedding {
            let mut out = vec![0.0f32; hidden];
            for (cb, token) in frame.iter().enumerate() {
                let idx =
                    cb * Self::CODEBOOK_VOCAB + (*token as usize).min(Self::CODEBOOK_VOCAB - 1);
                if idx >= meta.vocab_size {
                    return Err(Error::Execution(format!(
                        "LFM25AudioOnnxNode: audio embedding index {idx} out of range"
                    )));
                }
                let start = idx * meta.hidden_size;
                for (dst, src) in out
                    .iter_mut()
                    .zip(&weights[start..start + meta.hidden_size])
                {
                    *dst += *src;
                }
            }
            return Ok(out);
        }

        let audio_tokens: Vec<i64> = frame
            .iter()
            .enumerate()
            .map(|(cb, token)| (cb * Self::CODEBOOK_VOCAB) as i64 + *token)
            .collect();
        let input =
            Tensor::from_array(([1usize, Self::NUM_CODEBOOKS], audio_tokens)).map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: audio embedding tensor: {e}"))
            })?;
        let mut session = self.audio_embedding_session.lock();
        let outputs = session
            .run(ort::inputs!["audio_codes" => input])
            .map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: audio embedding session: {e}"))
            })?;
        let (_, values) = outputs["audio_embeds"]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: extract audio embeddings: {e}"))
            })?;
        let mut out = vec![0.0f32; hidden];
        for cb in 0..Self::NUM_CODEBOOKS {
            let row = &values[cb * hidden..(cb + 1) * hidden];
            for (dst, src) in out.iter_mut().zip(row) {
                *dst += *src;
            }
        }
        Ok(out)
    }

    fn init_decoder_cache(&self) -> L25Result<DecoderCache> {
        let mut values = HashMap::new();
        let head_dim = self.lfm.hidden_size / self.lfm.num_attention_heads.max(1);
        for (idx, layer_type) in self.lfm.layer_types.iter().enumerate() {
            if layer_type == "conv" {
                let tensor = Tensor::from_array((
                    [1usize, self.lfm.hidden_size, self.lfm.conv_L_cache],
                    vec![0.0f32; self.lfm.hidden_size * self.lfm.conv_L_cache],
                ))
                .map_err(|e| {
                    Error::Execution(format!("LFM25AudioOnnxNode: past_conv.{idx} tensor: {e}"))
                })?;
                values.insert(format!("past_conv.{idx}"), tensor.into_dyn());
            } else {
                let k = Tensor::<f32>::new(
                    &Allocator::default(),
                    [1usize, self.lfm.num_key_value_heads, 0usize, head_dim],
                )
                .map_err(|e| {
                    Error::Execution(format!(
                        "LFM25AudioOnnxNode: past_key_values.{idx}.key tensor: {e}"
                    ))
                })?;
                values.insert(format!("past_key_values.{idx}.key"), k.into_dyn());

                let v = Tensor::<f32>::new(
                    &Allocator::default(),
                    [1usize, self.lfm.num_key_value_heads, 0usize, head_dim],
                )
                .map_err(|e| {
                    Error::Execution(format!(
                        "LFM25AudioOnnxNode: past_key_values.{idx}.value tensor: {e}"
                    ))
                })?;
                values.insert(format!("past_key_values.{idx}.value"), v.into_dyn());
            }
        }
        Ok(DecoderCache {
            values,
            total_len: 0,
        })
    }

    fn clone_decoder_cache(cache: &DecoderCache) -> L25Result<DecoderCache> {
        let mut values = HashMap::with_capacity(cache.values.len());
        for (name, value) in &cache.values {
            let tensor_ref = value.downcast_ref::<DynTensorValueType>().map_err(|e| {
                Error::Execution(format!(
                    "LFM25AudioOnnxNode: cache tensor downcast for {name}: {e}"
                ))
            })?;
            let tensor = tensor_ref.try_upgrade().map_err(|_| {
                Error::Execution(format!(
                    "LFM25AudioOnnxNode: cache tensor upgrade for {name} failed"
                ))
            })?;
            values.insert(name.clone(), tensor.clone().into_dyn());
        }
        Ok(DecoderCache {
            values,
            total_len: cache.total_len,
        })
    }

    fn audio_prefix_cache(&self, prefix: &str, prefix_ids: &[u32]) -> L25Result<DecoderCache> {
        if let Some(cache) = self.prefix_caches.lock().get(prefix) {
            return Self::clone_decoder_cache(cache);
        }

        let embeds = self.text_embeddings(prefix_ids)?;
        let mut cache = self.init_decoder_cache()?;
        self.run_decoder_step(
            &embeds,
            prefix_ids.len(),
            &mut cache,
            DecoderReadout::Logits,
        )?;
        let cloned = Self::clone_decoder_cache(&cache)?;
        self.prefix_caches.lock().insert(prefix.to_string(), cache);
        Ok(cloned)
    }

    fn run_decoder_step(
        &self,
        embeddings: &[f32],
        input_seq_len: usize,
        cache: &mut DecoderCache,
        readout: DecoderReadout,
    ) -> L25Result<DecoderStepOutput> {
        let hidden = self.lfm.hidden_size;
        let embeds = Tensor::from_array(([1usize, input_seq_len, hidden], embeddings.to_vec()))
            .map_err(|e| {
                Error::Execution(format!(
                    "LFM25AudioOnnxNode: decoder embeddings tensor: {e}"
                ))
            })?;
        let total_len = cache.total_len + input_seq_len;
        let mask =
            Tensor::from_array(([1usize, total_len], vec![1i64; total_len])).map_err(|e| {
                Error::Execution(format!(
                    "LFM25AudioOnnxNode: decoder attention mask tensor: {e}"
                ))
            })?;
        let named: Vec<(String, ort::value::Value)> = vec![
            ("inputs_embeds".to_string(), embeds.into()),
            ("attention_mask".to_string(), mask.into()),
        ];
        let mut decoder_inputs = named
            .into_iter()
            .map(|(name, value)| (name, SessionInputValue::from(value)))
            .collect::<Vec<_>>();
        decoder_inputs.extend(
            cache
                .values
                .iter()
                .map(|(name, value)| (name.clone(), SessionInputValue::from(value))),
        );
        let mut selected_outputs = OutputSelector::no_default().with(match readout {
            DecoderReadout::Logits if self.decoder_token_output => "token_id",
            DecoderReadout::Logits => "logits",
            DecoderReadout::HiddenStates => "hidden_states",
        });
        for name in cache.values.keys() {
            let output_name = if let Some(idx) = name.strip_prefix("past_conv.") {
                format!("present_conv.{idx}")
            } else if let Some(rest) = name.strip_prefix("past_key_values.") {
                format!("present.{rest}")
            } else {
                continue;
            };
            selected_outputs = selected_outputs.with(output_name);
        }
        let run_options = RunOptions::new()
            .and_then(|opts| Ok(opts.with_outputs(selected_outputs)))
            .map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: decoder run options: {e}"))
            })?;
        let mut session = self.decoder.lock();
        let mut outputs: SessionOutputs = session
            .run_with_options(decoder_inputs, &run_options)
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: decoder run: {e}")))?;
        let token_id = if readout == DecoderReadout::Logits && self.decoder_token_output {
            let (_, token_ids) = outputs["token_id"]
                .try_extract_tensor::<i64>()
                .map_err(|e| {
                    Error::Execution(format!("LFM25AudioOnnxNode: decoder token extract: {e}"))
                })?;
            Some(token_ids.first().copied().ok_or_else(|| {
                Error::Execution("LFM25AudioOnnxNode: decoder token output was empty".into())
            })? as u32)
        } else if readout == DecoderReadout::Logits {
            let (_, logits) = outputs["logits"].try_extract_tensor::<f32>().map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: logits extract: {e}"))
            })?;
            let logits = logits.to_vec();
            let vocab = self.lfm.vocab_size;
            let logits_seq_len = logits.len() / vocab;
            let last = &logits[(logits_seq_len - 1) * vocab..logits_seq_len * vocab];
            let token = {
                let mut rng = self.rng.lock();
                sample_topk_temp(&mut *rng, last, self.text_temperature, 0) as u32
            };
            Some(token)
        } else {
            None
        };
        let hidden_states = if readout == DecoderReadout::HiddenStates {
            let (_, hidden_states) = outputs["hidden_states"]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    Error::Execution(format!("LFM25AudioOnnxNode: hidden_states extract: {e}"))
                })?;
            Some(hidden_states.to_vec())
        } else {
            None
        };

        for name in cache.values.keys().cloned().collect::<Vec<_>>() {
            let output_name = if let Some(idx) = name.strip_prefix("past_conv.") {
                format!("present_conv.{idx}")
            } else if let Some(rest) = name.strip_prefix("past_key_values.") {
                format!("present.{rest}")
            } else {
                continue;
            };
            let value = outputs.remove(&output_name).ok_or_else(|| {
                Error::Execution(format!(
                    "LFM25AudioOnnxNode: decoder output missing required cache tensor {output_name}"
                ))
            })?;
            cache.values.insert(name, value);
        }
        cache.total_len = total_len;

        Ok(DecoderStepOutput {
            token_id,
            hidden_states,
        })
    }

    fn sample_audio_codes(&self, hidden_last: &[f32]) -> L25Result<[i64; 8]> {
        let hidden = Tensor::from_array(([1usize, self.lfm.hidden_size], hidden_last.to_vec()))
            .map_err(|e| {
                Error::Execution(format!(
                    "LFM25AudioOnnxNode: depthformer hidden tensor: {e}"
                ))
            })?;
        if self.depthformer_unrolled {
            let mut session = self.depthformer.lock();
            let outputs = session
                .run(ort::inputs!["hidden_states" => hidden])
                .map_err(|e| {
                    Error::Execution(format!("LFM25AudioOnnxNode: unrolled depthformer run: {e}"))
                })?;
            let (_, audio_codes) =
                outputs["audio_codes"]
                    .try_extract_tensor::<i64>()
                    .map_err(|e| {
                        Error::Execution(format!(
                            "LFM25AudioOnnxNode: unrolled depthformer codes extract: {e}"
                        ))
                    })?;
            return audio_codes.to_vec().try_into().map_err(|codes: Vec<i64>| {
                Error::Execution(format!(
                    "LFM25AudioOnnxNode: unrolled depthformer returned {} codes, expected 8",
                    codes.len()
                ))
            });
        }
        let kv_cap = 6 * 1 * 8 * Self::NUM_CODEBOOKS * 32;
        let mut past_keys = Vec::<f32>::with_capacity(kv_cap);
        let mut past_values = Vec::<f32>::with_capacity(kv_cap);
        let mut depth_slices = vec![0.0f32; Self::NUM_CODEBOOKS * 1024];
        let mut prev_token = 0i64;
        let mut codes = [0i64; 8];
        let mut session = self.depthformer.lock();

        for step in 0..Self::NUM_CODEBOOKS {
            let depth_tensor = Tensor::from_array((
                [1usize, Self::NUM_CODEBOOKS, 1024usize],
                depth_slices.clone(),
            ))
            .map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: depth slices tensor: {e}"))
            })?;
            let step_tensor = Tensor::from_array((Vec::<usize>::new(), vec![step as i64]))
                .map_err(|e| {
                    Error::Execution(format!("LFM25AudioOnnxNode: depth step tensor: {e}"))
                })?;
            let prev_tensor = Tensor::from_array(([1usize], vec![prev_token])).map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: previous token tensor: {e}"))
            })?;
            let keys_tensor = if step == 0 {
                Tensor::<f32>::new(
                    &Allocator::default(),
                    [6usize, 1usize, 8usize, 0usize, 32usize],
                )
            } else {
                Tensor::from_array(([6usize, 1usize, 8usize, step, 32usize], past_keys.clone()))
            }
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: past keys tensor: {e}")))?;
            let values_tensor = if step == 0 {
                Tensor::<f32>::new(
                    &Allocator::default(),
                    [6usize, 1usize, 8usize, 0usize, 32usize],
                )
            } else {
                Tensor::from_array(([6usize, 1usize, 8usize, step, 32usize], past_values.clone()))
            }
            .map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: past values tensor: {e}"))
            })?;
            let seqlens_tensor =
                Tensor::from_array(([1usize], vec![step as i32])).map_err(|e| {
                    Error::Execution(format!("LFM25AudioOnnxNode: seqlens tensor: {e}"))
                })?;
            let total_tensor = Tensor::from_array((Vec::<usize>::new(), vec![(step + 1) as i32]))
                .map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: total seq len tensor: {e}"))
            })?;

            let mut selected_outputs =
                OutputSelector::no_default().with(if self.depthformer_token_output {
                    "token_id"
                } else {
                    "logits"
                });
            if step == 0 {
                selected_outputs = selected_outputs.with("depth_slices");
            }
            if step + 1 < Self::NUM_CODEBOOKS {
                selected_outputs = selected_outputs.with("new_keys").with("new_values");
            }
            let run_options = RunOptions::new()
                .and_then(|opts| Ok(opts.with_outputs(selected_outputs)))
                .map_err(|e| {
                    Error::Execution(format!("LFM25AudioOnnxNode: depthformer run options: {e}"))
                })?;

            let outputs = session
                .run_with_options(
                    ort::inputs![
                    "hidden_states" => hidden.clone(),
                    "depth_slices_in" => depth_tensor,
                    "step_idx" => step_tensor,
                    "prev_token" => prev_tensor,
                    "past_keys" => keys_tensor,
                    "past_values" => values_tensor,
                    "seqlens_k" => seqlens_tensor,
                    "total_seq_len" => total_tensor,
                    ],
                    &run_options,
                )
                .map_err(|e| {
                    Error::Execution(format!("LFM25AudioOnnxNode: depthformer run: {e}"))
                })?;

            let token = if self.depthformer_token_output {
                let (_, token_ids) =
                    outputs["token_id"]
                        .try_extract_tensor::<i64>()
                        .map_err(|e| {
                            Error::Execution(format!(
                                "LFM25AudioOnnxNode: depthformer token extract: {e}"
                            ))
                        })?;
                token_ids.first().copied().ok_or_else(|| {
                    Error::Execution(
                        "LFM25AudioOnnxNode: depthformer token output was empty".into(),
                    )
                })?
            } else {
                let (_, logits) = outputs["logits"].try_extract_tensor::<f32>().map_err(|e| {
                    Error::Execution(format!(
                        "LFM25AudioOnnxNode: depthformer logits extract: {e}"
                    ))
                })?;
                let mut rng = self.rng.lock();
                sample_topk_temp(&mut *rng, logits, self.audio_temperature, self.audio_top_k)
                    as i64
            };
            codes[step] = token;
            prev_token = token;

            if step == 0 {
                let (_, values) = outputs["depth_slices"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| {
                        Error::Execution(format!("LFM25AudioOnnxNode: depth slices extract: {e}"))
                    })?;
                depth_slices = values.to_vec();
            }
            if step + 1 < Self::NUM_CODEBOOKS {
                let (_, new_keys) =
                    outputs["new_keys"]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| {
                            Error::Execution(format!("LFM25AudioOnnxNode: new keys extract: {e}"))
                        })?;
                let (_, new_values) =
                    outputs["new_values"]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| {
                            Error::Execution(format!(
                                "LFM25AudioOnnxNode: new values extract: {e}"
                            ))
                        })?;
                past_keys = new_keys.to_vec();
                past_values = new_values.to_vec();
            }
        }

        Ok(codes)
    }

    fn decode_audio_codes(&self, frames: &[[i64; 8]]) -> L25Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let mut flat = Vec::with_capacity(Self::NUM_CODEBOOKS * frames.len());
        for cb in 0..Self::NUM_CODEBOOKS {
            for frame in frames {
                flat.push(frame[cb]);
            }
        }
        let input = Tensor::from_array(([1usize, Self::NUM_CODEBOOKS, frames.len()], flat))
            .map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: detokenizer input tensor: {e}"))
            })?;
        let mut session = self.audio_detokenizer.lock();
        let mut outputs = session
            .run(ort::inputs!["audio_codes" => input])
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: detokenizer run: {e}")))?;
        if let Some(audio_istft) = &self.audio_istft {
            let stft_features = outputs.remove("stft_features").ok_or_else(|| {
                Error::Execution(
                    "LFM25AudioOnnxNode: detokenizer missing stft_features output".into(),
                )
            })?;
            let mut istft = audio_istft.lock();
            let istft_outputs = istft
                .run(ort::inputs!["stft_features" => stft_features])
                .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: GPU ISTFT run: {e}")))?;
            let (_, waveform) = istft_outputs["waveform"]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    Error::Execution(format!("LFM25AudioOnnxNode: GPU ISTFT extract: {e}"))
                })?;
            return Ok(waveform.to_vec());
        }
        let (_, stft) = outputs["stft_features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: detokenizer extract: {e}"))
            })?;

        const N_FFT: usize = 1280;
        const HOP: usize = 320;
        const N_BINS: usize = N_FFT / 2 + 1;
        let frames_count = stft.len() / (N_BINS * 2);
        let mut waveform = vec![0.0f32; (frames_count.saturating_sub(1)) * HOP + N_FFT];
        let mut window_norm = vec![0.0f32; waveform.len()];

        for t in 0..frames_count {
            let base = t * (N_BINS * 2);
            let mut spectrum = vec![Complex32::new(0.0, 0.0); N_FFT];
            for bin in 0..N_BINS {
                let log_mag = stft[base + bin];
                let angle = stft[base + N_BINS + bin];
                let mag = log_mag.exp();
                spectrum[bin] = Complex32::from_polar(mag, angle);
            }
            for bin in 1..(N_BINS - 1) {
                spectrum[N_FFT - bin] = spectrum[bin].conj();
            }
            self.detokenizer_ifft.process(&mut spectrum);
            let offset = t * HOP;
            for i in 0..N_FFT {
                let sample = spectrum[i].re / N_FFT as f32 * self.detokenizer_window[i];
                waveform[offset + i] += sample;
                window_norm[offset + i] += self.detokenizer_window[i] * self.detokenizer_window[i];
            }
        }

        for (sample, norm) in waveform.iter_mut().zip(window_norm.iter()) {
            if *norm > 1e-8 {
                *sample /= *norm;
            }
        }
        let peak = waveform.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
        let pad = (N_FFT - HOP) / 2;
        let mut waveform = if waveform.len() > pad * 2 {
            waveform[pad..waveform.len() - pad].to_vec()
        } else {
            waveform
        };
        if peak > 0.0 {
            for sample in &mut waveform {
                *sample = *sample / peak * 0.9;
            }
        }
        Ok(waveform)
    }

    fn encode_audio(&self, samples: &[f32], sample_rate: u32) -> L25Result<Vec<f32>> {
        if sample_rate != self.mel.sample_rate {
            return Err(Error::Execution(format!(
                "LFM25AudioOnnxNode: ASR frontend expects {}Hz audio, got {}Hz",
                self.mel.sample_rate, sample_rate
            )));
        }
        let (mel_features, frame_count) = compute_mel(
            samples,
            &self.mel,
            &self.mel_filters,
            &self.mel_hann,
            &self.mel_fft,
        );
        let mel_tensor = Tensor::from_array(([1usize, frame_count, self.mel.n_mels], mel_features))
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: mel tensor: {e}")))?;
        let len_tensor = Tensor::from_array(([1usize], vec![frame_count as i64]))
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: mel length tensor: {e}")))?;
        let Some(audio_encoder) = &self.audio_encoder else {
            return Err(Error::Execution(
                "LFM25AudioOnnxNode: audio encoder is unavailable".into(),
            ));
        };
        let mut session = audio_encoder.lock();
        let outputs = session
            .run(ort::inputs![
                "mel_spectrogram" => mel_tensor,
                "mel_lengths" => len_tensor,
            ])
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: audio encoder run: {e}")))?;
        let (_, embeddings) = outputs["audio_embeddings"]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                Error::Execution(format!("LFM25AudioOnnxNode: audio embeddings extract: {e}"))
            })?;
        Ok(embeddings.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Mel filterbank + FFT helpers
// ---------------------------------------------------------------------------

fn hz_to_mel(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn mel_filterbank(config: &MelConfig) -> Vec<Vec<(usize, f32)>> {
    let bins = config.n_fft / 2 + 1;
    let mel_min = hz_to_mel(config.fmin);
    let mel_max = hz_to_mel(config.fmax);
    let points: Vec<f32> = (0..config.n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (config.n_mels + 1) as f32))
        .collect();
    let fft_freqs: Vec<f32> = (0..bins)
        .map(|i| i as f32 * config.sample_rate as f32 / config.n_fft as f32)
        .collect();
    let mut filters = vec![Vec::new(); config.n_mels];
    for m in 0..config.n_mels {
        let left = points[m];
        let center = points[m + 1];
        let right = points[m + 2];
        // Slaney/librosa area normalization: each triangular filter has
        // weight `2.0 / (right - left)` applied so the total area
        // integrates to ~1.
        let slaney = 2.0 / (right - left).max(1e-6);
        for (bin, freq) in fft_freqs.iter().enumerate() {
            let weight = if *freq >= left && *freq <= center {
                slaney * (*freq - left) / (center - left).max(1e-6)
            } else if *freq > center && *freq <= right {
                slaney * (right - *freq) / (right - center).max(1e-6)
            } else {
                0.0
            };
            if weight != 0.0 {
                filters[m].push((bin, weight));
            }
        }
    }
    filters
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / (size as f32 - 1.0)).cos())
        .collect()
}

fn compute_mel(
    samples: &[f32],
    config: &MelConfig,
    filters: &[Vec<(usize, f32)>],
    hann: &[f32],
    fft: &Arc<dyn Fft<f32>>,
) -> (Vec<f32>, usize) {
    let mut emphasized = Vec::with_capacity(samples.len());
    if let Some((&first, rest)) = samples.split_first() {
        emphasized.push(first);
        for idx in 0..rest.len() {
            emphasized.push(rest[idx] - config.preemph * samples[idx]);
        }
    }
    let pad = config.n_fft / 2;
    let mut padded = vec![0.0f32; pad];
    padded.extend_from_slice(&emphasized);
    padded.extend(std::iter::repeat(0.0f32).take(pad));
    let frame_count = 1 + padded.len().saturating_sub(config.n_fft) / config.hop_length;
    let pad_left = (config.n_fft - config.win_length) / 2;
    let mut mel = vec![0.0f32; frame_count * config.n_mels];
    let mut frame = vec![Complex32::new(0.0, 0.0); config.n_fft];
    let mut power = vec![0.0f32; config.n_fft / 2 + 1];
    for frame_idx in 0..frame_count {
        let start = frame_idx * config.hop_length;
        frame.fill(Complex32::new(0.0, 0.0));
        for i in 0..config.win_length {
            frame[pad_left + i].re = padded[start + pad_left + i] * hann[i];
        }
        fft.process(&mut frame);
        for (dst, complex) in power.iter_mut().zip(frame.iter()) {
            *dst = complex.norm_sqr();
        }
        for mel_idx in 0..config.n_mels {
            let energy: f32 = filters[mel_idx]
                .iter()
                .map(|(bin, weight)| weight * power[*bin])
                .sum();
            mel[frame_idx * config.n_mels + mel_idx] = (energy + config.log_zero_guard).ln();
        }
    }

    // Per-feature normalization (NEMO `normalize: "per_feature"`).
    if frame_count > 1 {
        const NORM_EPS: f32 = 1.0e-5;
        for m in 0..config.n_mels {
            let mut sum = 0.0f32;
            for t in 0..frame_count {
                sum += mel[t * config.n_mels + m];
            }
            let mean = sum / frame_count as f32;
            let mut var_sum = 0.0f32;
            for t in 0..frame_count {
                let d = mel[t * config.n_mels + m] - mean;
                var_sum += d * d;
            }
            let std = (var_sum / (frame_count - 1) as f32).sqrt() + NORM_EPS;
            for t in 0..frame_count {
                mel[t * config.n_mels + m] = (mel[t * config.n_mels + m] - mean) / std;
            }
        }
    }

    (mel, frame_count)
}

fn read_meta(path: &Path) -> L25Result<EmbeddingMeta> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| {
        Error::Execution(format!(
            "LFM25AudioOnnxNode: invalid {}: {e}",
            path.display()
        ))
    })
}

fn read_f32_blob(path: &Path) -> L25Result<Vec<f32>> {
    let mut file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.len() % 4 != 0 {
        return Err(Error::Execution(format!(
            "LFM25AudioOnnxNode: {} byte length {} is not aligned to f32",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn load_session(path: PathBuf, device: &str) -> L25Result<Session> {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");
    let use_cuda = device.starts_with("cuda");

    if use_cuda {
        let device_id = device
            .strip_prefix("cuda:")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        tracing::info!(
            graph = file_name,
            device_id,
            "LFM25AudioOnnxNode: loading with CUDA EP"
        );
        // CUDA EP — kept at defaults. `with_tf32(true)` and
        // `with_conv1d_pad_to_nc1d(true)` both made detokenizer 50× slower
        // and decoder 15× slower in bench (2026-05-13); the default
        // `cudnn_conv_algo_search=EXHAUSTIVE` picks the best kernels.
        let cuda_providers: Vec<ExecutionProviderDispatch> = vec![
            CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build(),
            CPUExecutionProvider::default().build(),
        ];

        let verbose_log = std::env::var_os("LFM25_ORT_VERBOSE").is_some();
        let mut builder = Session::builder()
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: ORT builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: ORT optimization: {e}")))?
            .with_execution_providers(cuda_providers)
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: ORT providers: {e}")))?;
        if verbose_log {
            builder = builder
                .with_log_level(ort::logging::LogLevel::Info)
                .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: log_level: {e}")))?;
        }
        match builder.commit_from_file(&path) {
            Ok(session) => {
                tracing::info!(
                    graph = file_name,
                    ep = "CUDA",
                    "LFM25AudioOnnxNode: session loaded successfully"
                );
                return Ok(session);
            }
            Err(e) => {
                tracing::warn!(
                    graph = file_name,
                    ep = "CUDA",
                    error = %e,
                    "LFM25AudioOnnxNode: CUDA EP failed, falling back to CPU"
                );
            }
        }
    }

    let session = Session::builder()
        .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: ORT builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: ORT optimization: {e}")))?
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode: ORT providers: {e}")))?
        .commit_from_file(&path)
        .map_err(|e| {
            Error::Execution(format!(
                "LFM25AudioOnnxNode: failed to load {}: {e}",
                path.display()
            ))
        })?;
    tracing::info!(
        graph = file_name,
        ep = "CPU",
        "LFM25AudioOnnxNode: session loaded"
    );
    Ok(session)
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

#[derive(Default)]
struct OrtLFM25AudioBackend {
    runtime: Mutex<Option<Arc<OrtLFM25Runtime>>>,
}

impl OrtLFM25AudioBackend {
    fn runtime(&self, config: &LFM25AudioOnnxConfig) -> L25Result<Arc<OrtLFM25Runtime>> {
        let mut guard = self.runtime.lock();
        if guard.is_none() {
            *guard = Some(Arc::new(OrtLFM25Runtime::load(config)?));
        }
        Ok(guard.as_ref().cloned().expect("runtime inserted"))
    }

    fn onnx_dir(config: &LFM25AudioOnnxConfig) -> PathBuf {
        let nested = config.model_dir.join("onnx");
        if nested.is_dir() {
            nested
        } else {
            config.model_dir.clone()
        }
    }

    fn required_files(config: &LFM25AudioOnnxConfig) -> Vec<PathBuf> {
        let suffix = config.precision.suffix();
        let onnx_dir = Self::onnx_dir(config);
        let names = [
            format!("decoder{suffix}.onnx"),
            format!("audio_encoder{suffix}.onnx"),
            format!("audio_embedding{suffix}.onnx"),
            format!("audio_detokenizer{suffix}.onnx"),
            format!("vocoder_depthformer{suffix}.onnx"),
            "embed_tokens.bin".to_string(),
            "embed_tokens.json".to_string(),
            "audio_embedding.bin".to_string(),
            "audio_embedding.json".to_string(),
            "mel_config.json".to_string(),
        ];
        names.into_iter().map(|name| onnx_dir.join(name)).collect()
    }

    fn split_data_sidecars_present(path: &Path) -> bool {
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            return false;
        };
        let Some(parent) = path.parent() else {
            return false;
        };
        let prefix = format!("{file_name}_data");
        fs::read_dir(parent)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .any(|name| name == prefix || name.starts_with(&(prefix.clone() + "_")))
    }
}

impl LFM25AudioBackend for OrtLFM25AudioBackend {
    fn validate_bundle(&self, config: &LFM25AudioOnnxConfig) -> L25Result<()> {
        let mut missing: Vec<String> = Self::required_files(config)
            .into_iter()
            .filter(|path| !path.exists())
            .map(|path| path.display().to_string())
            .collect();

        let onnx_dir = Self::onnx_dir(config);
        for stem in [
            format!("decoder{}.onnx", config.precision.suffix()),
            format!("audio_encoder{}.onnx", config.precision.suffix()),
            format!("audio_embedding{}.onnx", config.precision.suffix()),
            format!("audio_detokenizer{}.onnx", config.precision.suffix()),
            format!("vocoder_depthformer{}.onnx", config.precision.suffix()),
        ] {
            let path = onnx_dir.join(stem);
            if path.exists() && !Self::split_data_sidecars_present(&path) {
                missing.push(format!(
                    "{} external data sidecar(s): expected {}_data*",
                    path.display(),
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("model.onnx")
                ));
            }
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(Error::Execution(format!(
                "LFM25AudioOnnxNode: model bundle is incomplete; missing {}",
                missing.join(", ")
            )))
        }
    }

    fn prewarm(&self, config: &LFM25AudioOnnxConfig) -> L25Result<()> {
        let runtime = self.runtime(config)?;
        let mut warm_config = config.clone();
        warm_config.max_new_tokens = warm_config.max_new_tokens.min(4).max(1);
        warm_config.audio_batch_size = warm_config.audio_batch_size.max(1);
        warm_config.first_chunk_audio_batch_size =
            warm_config.first_chunk_audio_batch_size.or(Some(1));

        let mut discard = |_item: RuntimeData| Ok(());
        match config.mode {
            LFM25AudioMode::Asr | LFM25AudioMode::Interleaved => {
                let silence = vec![0.0f32; (config.input_sample_rate as usize / 4).max(160)];
                let _ = generate_from_audio_streaming(
                    runtime,
                    &silence,
                    config.input_sample_rate,
                    config.mode,
                    &warm_config,
                    &mut discard,
                )?;
            }
            LFM25AudioMode::Tts => {
                let _ = generate_from_text_streaming(
                    runtime,
                    "warmup",
                    LFM25AudioMode::Tts,
                    &warm_config,
                    &mut discard,
                )?;
            }
        }
        Ok(())
    }

    fn generate_streaming(
        &self,
        input: LFM25Input,
        config: &LFM25AudioOnnxConfig,
        emit: &mut dyn FnMut(RuntimeData) -> L25Result<()>,
    ) -> L25Result<usize> {
        let runtime = self.runtime(config)?;

        match input {
            LFM25Input::Text { text, mode } => {
                generate_from_text_streaming(runtime, &text, mode, config, emit)
            }
            LFM25Input::Audio {
                samples,
                sample_rate,
                mode,
            } => generate_from_audio_streaming(runtime, &samples, sample_rate, mode, config, emit),
        }
    }
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Sample a token id from raw logits with temperature + optional top-k.
/// Mirrors the reference `_sample` in `liquidonnx/lfm2_audio/infer.py`.
fn sample_topk_temp(rng: &mut impl Rng, logits: &[f32], temperature: f32, top_k: usize) -> usize {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let inv_temp = 1.0 / temperature;
    let n = logits.len();
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l * inv_temp).collect();
    if top_k > 0 && top_k < n {
        let mut sorted: Vec<f32> = scaled.clone();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let cutoff = sorted[top_k - 1];
        for v in scaled.iter_mut() {
            if *v < cutoff {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    let max = scaled
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    if !max.is_finite() {
        return argmax(logits);
    }
    let mut sum = 0.0f32;
    let mut probs: Vec<f32> = Vec::with_capacity(n);
    for &v in &scaled {
        let p = (v - max).exp();
        probs.push(p);
        sum += p;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return argmax(logits);
    }
    let r: f32 = rng.gen();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p / sum;
        if r < cum {
            return i;
        }
    }
    n - 1
}

// ---------------------------------------------------------------------------
// Generation loops
// ---------------------------------------------------------------------------

fn generate_from_text_streaming(
    runtime: Arc<OrtLFM25Runtime>,
    text: &str,
    mode: LFM25AudioMode,
    config: &LFM25AudioOnnxConfig,
    emit: &mut dyn FnMut(RuntimeData) -> L25Result<()>,
) -> L25Result<usize> {
    let system_prompt = effective_system_prompt(mode, config);
    let prompt = match mode {
        LFM25AudioMode::Tts => format!(
            "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, text
        ),
        LFM25AudioMode::Interleaved => format!(
            "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, text
        ),
        LFM25AudioMode::Asr => {
            return Err(Error::Execution(
                "LFM25AudioOnnxNode: text input is not valid in ASR mode".into(),
            ))
        }
    };

    let ids = runtime.encode(&prompt)?;
    let mut embeds = runtime.text_embeddings(&ids)?;
    let mut generated = Vec::<u32>::new();
    let mut in_audio = false;
    let mut cache = runtime.init_decoder_cache()?;
    let mut input_seq_len = ids.len();
    let mut emitted = 0usize;

    let mut audio_batch = Vec::<[i64; 8]>::new();
    let first_chunk_size = config
        .first_chunk_audio_batch_size
        .unwrap_or(config.audio_batch_size);
    let mut first_chunk_emitted = false;
    let mut emitted_audio_samples = 0usize;

    for _ in 0..config.max_new_tokens.min(512) {
        if in_audio {
            let decoder_output = runtime.run_decoder_step(
                &embeds,
                input_seq_len,
                &mut cache,
                DecoderReadout::HiddenStates,
            )?;
            let hidden_states = decoder_output.hidden_states.ok_or_else(|| {
                Error::Execution(
                    "LFM25AudioOnnxNode: decoder omitted hidden states in audio mode".into(),
                )
            })?;
            let hidden = runtime.lfm.hidden_size;
            let hidden_seq_len = hidden_states.len() / hidden;
            let start = (hidden_seq_len - 1) * hidden;
            let mut frame = runtime.sample_audio_codes(&hidden_states[start..start + hidden])?;
            if frame[0] == OrtLFM25Runtime::END_OF_AUDIO_TOKEN {
                break;
            }
            for token in &mut frame {
                *token = (*token).min(2047);
            }
            audio_batch.push(frame);

            let threshold = if first_chunk_emitted {
                config.audio_batch_size
            } else {
                first_chunk_size
            };
            if audio_batch.len() >= threshold {
                let waveform = runtime.decode_audio_codes(&audio_batch)?;
                let pts_us =
                    (emitted_audio_samples as u64 * 1_000_000) / config.output_sample_rate as u64;
                emitted_audio_samples += waveform.len();
                emit(RuntimeData::Audio {
                    samples: AudioSamples::from(waveform),
                    sample_rate: config.output_sample_rate,
                    channels: 1,
                    stream_id: None,
                    timestamp_us: Some(pts_us),
                    arrival_ts_us: None,
                    metadata: Some(serde_json::json!({"pts_us": pts_us})),
                })?;
                emitted += 1;
                audio_batch.clear();
                first_chunk_emitted = true;
            }

            embeds = runtime.audio_embeddings_sum(&frame)?;
            input_seq_len = 1;
        } else {
            let decoder_output = runtime.run_decoder_step(
                &embeds,
                input_seq_len,
                &mut cache,
                DecoderReadout::Logits,
            )?;
            let token = decoder_output.token_id.ok_or_else(|| {
                Error::Execution("LFM25AudioOnnxNode: decoder omitted token id in text mode".into())
            })?;
            if token == OrtLFM25Runtime::IM_END_TOKEN {
                break;
            }
            generated.push(token);
            embeds = runtime.text_embeddings(&[token])?;
            input_seq_len = 1;
            if token == OrtLFM25Runtime::AUDIO_START_TOKEN
                || token == OrtLFM25Runtime::TEXT_END_TOKEN
            {
                let text_out = runtime.decode(&generated)?;
                if !text_out.trim().is_empty() {
                    emit(RuntimeData::Text(text_out))?;
                    emitted += 1;
                }
                in_audio = true;
            }
        }
    }

    if !in_audio && !generated.is_empty() {
        let text_out = runtime.decode(&generated)?;
        if !text_out.trim().is_empty() {
            emit(RuntimeData::Text(text_out))?;
            emitted += 1;
        }
    }
    if !audio_batch.is_empty() {
        let waveform = runtime.decode_audio_codes(&audio_batch)?;
        let pts_us = (emitted_audio_samples as u64 * 1_000_000) / config.output_sample_rate as u64;
        emit(RuntimeData::Audio {
            samples: AudioSamples::from(waveform),
            sample_rate: config.output_sample_rate,
            channels: 1,
            stream_id: None,
            timestamp_us: Some(pts_us),
            arrival_ts_us: None,
            metadata: Some(serde_json::json!({"pts_us": pts_us})),
        })?;
        emitted += 1;
    }
    Ok(emitted)
}

fn generate_from_audio_streaming(
    runtime: Arc<OrtLFM25Runtime>,
    samples: &[f32],
    sample_rate: u32,
    mode: LFM25AudioMode,
    config: &LFM25AudioOnnxConfig,
    emit: &mut dyn FnMut(RuntimeData) -> L25Result<()>,
) -> L25Result<usize> {
    let turn_started = Instant::now();
    let audio_frontend_started = Instant::now();
    let audio_embeddings = runtime.encode_audio(samples, sample_rate)?;
    let audio_frontend_ms = audio_frontend_started.elapsed().as_millis() as u64;
    let hidden = runtime.lfm.hidden_size;
    let audio_seq = audio_embeddings.len() / hidden;
    let system_prompt = effective_system_prompt(mode, config);
    let (prefix, suffix) = match mode {
        LFM25AudioMode::Asr => (
            format!(
                "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
                system_prompt
            ),
            "<|im_end|>\n<|im_start|>assistant\n".to_string(),
        ),
        LFM25AudioMode::Interleaved => (
            format!(
                "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
                system_prompt
            ),
            "<|im_end|>\n<|im_start|>assistant\n".to_string(),
        ),
        LFM25AudioMode::Tts => {
            return Err(Error::Execution(
                "LFM25AudioOnnxNode: audio input is not valid in TTS mode".into(),
            ))
        }
    };
    let prefix_ids = runtime.encode(&prefix)?;
    let suffix_ids = runtime.encode(&suffix)?;
    let mut cache = runtime.audio_prefix_cache(&prefix, &prefix_ids)?;
    let mut embeds = audio_embeddings;
    embeds.extend_from_slice(&runtime.text_embeddings(&suffix_ids)?);
    let mut generated = Vec::<u32>::new();
    let mut in_audio = false;
    let mut input_seq_len = audio_seq + suffix_ids.len();
    let mut emitted = 0usize;
    let mut decoder_ms = 0u64;
    let mut depthformer_ms = 0u64;
    let mut detokenizer_ms = 0u64;
    let mut audio_embed_ms = 0u64;
    let mut decoder_steps_before_audio = 0usize;
    let mut depthformer_frames_before_first_emit = 0usize;
    let mut total_audio_frames = 0usize;
    let mut first_audio_logged = false;
    let _ = config.audio_first_interleaved;

    let mut audio_batch = Vec::<[i64; 8]>::new();
    let first_chunk_size = config
        .first_chunk_audio_batch_size
        .unwrap_or(config.audio_batch_size);
    let mut first_chunk_emitted = false;
    let mut emitted_audio_samples = 0usize;
    let interleaved = mode == LFM25AudioMode::Interleaved;
    let n_text_run = OrtLFM25Runtime::INTERLEAVED_TEXT_RUN;
    let n_audio_run = OrtLFM25Runtime::INTERLEAVED_AUDIO_RUN;
    let mut modality_left = n_text_run;
    let mut text_done = false;

    for _ in 0..config.max_new_tokens.min(256) {
        if interleaved && modality_left > 0 {
            modality_left -= 1;
        }
        let decoder_started = Instant::now();
        let decoder_output = runtime.run_decoder_step(
            &embeds,
            input_seq_len,
            &mut cache,
            if in_audio {
                DecoderReadout::HiddenStates
            } else {
                DecoderReadout::Logits
            },
        )?;
        decoder_ms += decoder_started.elapsed().as_millis() as u64;
        if in_audio {
            let hidden_states = decoder_output.hidden_states.ok_or_else(|| {
                Error::Execution(
                    "LFM25AudioOnnxNode: decoder omitted hidden states in audio mode".into(),
                )
            })?;
            let hidden_seq_len = hidden_states.len() / hidden;
            let start = (hidden_seq_len - 1) * hidden;
            let depthformer_started = Instant::now();
            let mut frame = runtime.sample_audio_codes(&hidden_states[start..start + hidden])?;
            depthformer_ms += depthformer_started.elapsed().as_millis() as u64;
            if !first_chunk_emitted {
                depthformer_frames_before_first_emit += 1;
            }

            if interleaved && modality_left == 0 && !text_done {
                in_audio = false;
                modality_left = n_text_run;
            }

            let is_eoa = frame[0] == OrtLFM25Runtime::END_OF_AUDIO_TOKEN;
            if is_eoa {
                for token in &mut frame {
                    *token = OrtLFM25Runtime::END_OF_AUDIO_TOKEN;
                }
                if interleaved {
                    in_audio = false;
                    modality_left = n_text_run;
                }
            } else {
                for token in &mut frame {
                    *token = (*token).min(2047);
                }
                audio_batch.push(frame);
                total_audio_frames += 1;
            }

            let threshold = if first_chunk_emitted {
                config.audio_batch_size
            } else {
                first_chunk_size
            };
            if audio_batch.len() >= threshold {
                let detokenizer_started = Instant::now();
                let waveform = runtime.decode_audio_codes(&audio_batch)?;
                detokenizer_ms += detokenizer_started.elapsed().as_millis() as u64;
                let pts_us =
                    (emitted_audio_samples as u64 * 1_000_000) / config.output_sample_rate as u64;
                emitted_audio_samples += waveform.len();
                emit(RuntimeData::Audio {
                    samples: AudioSamples::from(waveform),
                    sample_rate: config.output_sample_rate,
                    channels: 1,
                    stream_id: None,
                    timestamp_us: Some(pts_us),
                    arrival_ts_us: None,
                    metadata: Some(serde_json::json!({"pts_us": pts_us})),
                })?;
                emitted += 1;
                audio_batch.clear();
                first_chunk_emitted = true;
                if !first_audio_logged {
                    tracing::info!(
                        node = "LFM25AudioOnnxNode",
                        total_ms = turn_started.elapsed().as_millis() as u64,
                        audio_frontend_ms,
                        decoder_ms,
                        depthformer_ms,
                        detokenizer_ms,
                        audio_embed_ms,
                        decoder_steps_before_audio,
                        depthformer_frames_before_first_emit,
                        first_chunk_size,
                        "LFM25 first audio emitted"
                    );
                    first_audio_logged = true;
                }
            }

            let embed_started = Instant::now();
            embeds = runtime.audio_embeddings_sum(&frame)?;
            audio_embed_ms += embed_started.elapsed().as_millis() as u64;
            input_seq_len = 1;
        } else {
            decoder_steps_before_audio += 1;
            let token = decoder_output.token_id.ok_or_else(|| {
                Error::Execution("LFM25AudioOnnxNode: decoder omitted token id in text mode".into())
            })?;
            if token == OrtLFM25Runtime::IM_END_TOKEN {
                break;
            }
            generated.push(token);
            embeds = runtime.text_embeddings(&[token])?;
            input_seq_len = 1;
            if interleaved && token == OrtLFM25Runtime::TEXT_END_TOKEN {
                text_done = true;
            }
            if interleaved && (modality_left == 0 || text_done) {
                let text_out = runtime.decode(&generated)?;
                if !text_out.trim().is_empty() {
                    emit(RuntimeData::Text(text_out))?;
                    emitted += 1;
                }
                in_audio = true;
                modality_left = n_audio_run;
            }
        }
    }

    if !in_audio && !generated.is_empty() {
        let text_out = runtime.decode(&generated)?;
        if !text_out.trim().is_empty() {
            emit(RuntimeData::Text(text_out))?;
            emitted += 1;
        }
    }
    if !audio_batch.is_empty() {
        let detokenizer_started = Instant::now();
        let waveform = runtime.decode_audio_codes(&audio_batch)?;
        detokenizer_ms += detokenizer_started.elapsed().as_millis() as u64;
        let pts_us = (emitted_audio_samples as u64 * 1_000_000) / config.output_sample_rate as u64;
        emit(RuntimeData::Audio {
            samples: AudioSamples::from(waveform),
            sample_rate: config.output_sample_rate,
            channels: 1,
            stream_id: None,
            timestamp_us: Some(pts_us),
            arrival_ts_us: None,
            metadata: Some(serde_json::json!({"pts_us": pts_us})),
        })?;
        emitted += 1;
    } else if mode == LFM25AudioMode::Interleaved && audio_seq > 0 && !in_audio {
        tracing::debug!(
            audio_seq,
            "LFM25AudioOnnxNode: interleaved audio input completed without entering audio generation"
        );
    }
    tracing::info!(
        node = "LFM25AudioOnnxNode",
        total_ms = turn_started.elapsed().as_millis() as u64,
        audio_frontend_ms,
        decoder_ms,
        depthformer_ms,
        detokenizer_ms,
        audio_embed_ms,
        decoder_steps_before_audio,
        total_audio_frames,
        emitted,
        first_chunk_size,
        "LFM25 audio turn completed"
    );
    Ok(emitted)
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

pub struct LFM25AudioOnnxNode {
    config: LFM25AudioOnnxConfig,
    backend: Arc<dyn LFM25AudioBackend>,
    sessions: Mutex<HashMap<String, SessionState>>,
}

impl LFM25AudioOnnxNode {
    /// Construct with default ORT-backed implementation.
    pub fn new(config: LFM25AudioOnnxConfig) -> Self {
        Self {
            config,
            backend: Arc::new(OrtLFM25AudioBackend::default()),
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Legacy constructor — kept for back-compat with existing call sites.
    pub fn with_config(config: LFM25AudioOnnxConfig) -> Self {
        Self::new(config)
    }

    fn session_key(session_id: Option<String>) -> String {
        session_id.unwrap_or_else(|| "default".to_string())
    }

    fn handle_aux(&self, data: &RuntimeData) -> L25Result<bool> {
        let RuntimeData::Json(value) = data else {
            return Ok(false);
        };
        let Some(port) = value.get(AUX_PORT_ENVELOPE_KEY).and_then(Value::as_str) else {
            return Ok(false);
        };
        let payload = value.get("payload").cloned().unwrap_or(Value::Null);
        let mut sessions = self.sessions.lock();
        match port {
            "context" => {
                let text = payload
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                for state in sessions.values_mut() {
                    state.context = text.clone();
                    state.turn_count = 0;
                }
                sessions.clear();
            }
            "system_prompt" => {
                let prompt = payload
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                for state in sessions.values_mut() {
                    state.system_prompt = prompt.clone();
                    state.turn_count = 0;
                }
                sessions.clear();
            }
            "reset" => {
                sessions.clear();
            }
            "barge_in" => {
                // The router normally intercepts this before it reaches the node.
                // Keep this branch for direct/unit-test callers.
            }
            _ => {}
        }
        Ok(true)
    }

    fn build_input(&self, data: RuntimeData) -> L25Result<LFM25Input> {
        match data {
            RuntimeData::Audio {
                samples,
                sample_rate,
                ..
            } => {
                if !matches!(
                    self.config.mode,
                    LFM25AudioMode::Asr | LFM25AudioMode::Interleaved
                ) {
                    return Err(Error::Execution(
                        "LFM25AudioOnnxNode: audio input requires asr or interleaved mode".into(),
                    ));
                }
                if sample_rate != self.config.input_sample_rate {
                    return Err(Error::Execution(format!(
                        "LFM25AudioOnnxNode: expected {}Hz audio input, got {}Hz",
                        self.config.input_sample_rate, sample_rate
                    )));
                }
                Ok(LFM25Input::Audio {
                    samples: samples.to_vec(),
                    sample_rate,
                    mode: self.config.mode,
                })
            }
            RuntimeData::Text(text) => {
                if !matches!(
                    self.config.mode,
                    LFM25AudioMode::Tts | LFM25AudioMode::Interleaved
                ) {
                    return Err(Error::Execution(
                        "LFM25AudioOnnxNode: text input requires tts or interleaved mode".into(),
                    ));
                }
                if text.trim().is_empty() {
                    return Err(Error::Execution(
                        "LFM25AudioOnnxNode: text input must not be empty".into(),
                    ));
                }
                Ok(LFM25Input::Text {
                    text,
                    mode: self.config.mode,
                })
            }
            other => Err(Error::Execution(format!(
                "LFM25AudioOnnxNode expects Audio, Text, or aux Json input; got {}",
                other.data_type()
            ))),
        }
    }
}

#[async_trait]
impl AsyncStreamingNode for LFM25AudioOnnxNode {
    fn node_type(&self) -> &str {
        "LFM25AudioOnnxNode"
    }

    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        let input = self.build_input(data)?;
        let mut last = None;
        self.backend
            .generate_streaming(input, &self.config, &mut |item| {
                last = Some(item);
                Ok(())
            })?;
        last.ok_or_else(|| Error::Execution("LFM25AudioOnnxNode produced no outputs".into()))
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        if self.handle_aux(&data)? {
            return Ok(0);
        }

        let sid = Self::session_key(session_id);
        {
            let mut sessions = self.sessions.lock();
            let state = sessions.entry(sid).or_insert_with(|| SessionState {
                context: String::new(),
                system_prompt: self.config.system_prompt.clone(),
                turn_count: 0,
            });
            state.turn_count += 1;
        }

        let input = self.build_input(data)?;

        // Run the sync ONNX generation on a blocking thread and shuttle
        // each emit through a tokio channel. Two reasons:
        //   1. `backend.generate_streaming` is a tight ORT FFI loop with
        //      zero `.await`s — running it inline pins the tokio worker
        //      for the entire turn (~600 ms on a 4090).
        //   2. The async parent below awaits `emit_rx.recv()` per frame,
        //      yielding to the runtime between emits so the host's
        //      streaming fan-out can drain the just-emitted frame BEFORE
        //      the model produces the next one.
        let backend = Arc::clone(&self.backend);
        let config = self.config.clone();
        let (emit_tx, mut emit_rx) = tokio::sync::mpsc::channel::<RuntimeData>(64);

        let gen_handle = tokio::task::spawn_blocking(move || -> L25Result<usize> {
            let mut emitted = 0usize;
            backend.generate_streaming(input, &config, &mut |item| {
                emit_tx
                    .blocking_send(item)
                    .map_err(|_| Error::Execution("LFM25AudioOnnxNode: emit channel closed".into()))?;
                emitted += 1;
                Ok(())
            })?;
            Ok(emitted)
        });

        while let Some(item) = emit_rx.recv().await {
            callback(item)?;
        }

        let emitted = gen_handle
            .await
            .map_err(|e| Error::Execution(format!("LFM25AudioOnnxNode generation join: {}", e)))??;

        callback(RuntimeData::Text(TEXT_END.to_string()))?;
        callback(RuntimeData::Text(AUDIO_END.to_string()))?;
        Ok(emitted + 2)
    }

    async fn process_control_message(
        &self,
        message: RuntimeData,
        _session_id: Option<String>,
    ) -> Result<bool, Error> {
        self.handle_aux(&message)
    }
}

// ---------------------------------------------------------------------------
// Factory + plugin registration
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct LFM25AudioOnnxNodeFactory;

impl FfiNodeFactory for LFM25AudioOnnxNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("LFM25AudioOnnxNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let cfg: LFM25AudioOnnxConfig =
            serde_json::from_str(params.as_str()).unwrap_or_default();
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(LFM25AudioOnnxNode::new(cfg)),
            TD_Opaque,
        ))
    }
}

remotemedia_plugin_sdk::plugin_export!(LFM25AudioOnnxNodeFactory);
