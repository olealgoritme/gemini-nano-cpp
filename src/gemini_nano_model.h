// What Gemini Nano v3 ("v3Nano") is, as far as can be established without
// decrypting anything.
//
// Every value is tagged with how it is known:
//   [measured]  read from the model files' open container layout
//               (scripts/inspect-model.py)
//   [api]       reported by Chrome's model library through its API
//               (gnano --info, --dump-tokenizer, --tokenize)
//   [verified]  taken from the public Gemma 3n E2B config and confirmed against
//               this model by scripts/verify-model.py
//   [public]    from the public Gemma 3n E2B config, consistent with this model
//               but not directly testable (the section headers that describe
//               the network are encrypted)
//
// Conclusion: Gemini Nano v3 is a Gemma 3n E2B-architecture model with the
// same tokenizer (plus ~6,200 extra internal tokens), not a different design.

#ifndef GEMINI_NANO_MODEL_H_
#define GEMINI_NANO_MODEL_H_

#include <stddef.h>
#include <stdint.h>

namespace nano::model {

// ---------------------------------------------------------------------------
// Model files (Chrome component "Optimization Guide On Device Model")

struct Variant {
  const char* version;        // [measured] manifest.json "version"
  const char* base_version;   // [measured] BaseModelSpec.version
  const char* backend;        // "gpu" or "cpu"
  uint64_t weights_bytes;     // [measured] weights.bin
  bool readable_container;    // [measured] ZIP sections visible?
};

inline constexpr Variant kGpu = {"2025.8.8.1141", "2025.06.30.1229", "gpu",
                                 4'269'932'544, false};
inline constexpr Variant kCpu = {"2025.8.21.1028", "2025.08.14.1358", "cpu",
                                 2'862'920'655, true};

// [measured] The CPU weights.bin is a ZIP of stored (uncompressed) sections.
// Each section starts with an encrypted header (a TFLite flatbuffer would
// show "TFL3" at byte 4); the bulk data after it is not encrypted. The CPU
// model also ships cache.bin (1,409,783,784 bytes), a pre-packed weight cache
// for the CPU kernels. The GPU weights.bin is opaque from its first byte.
struct Section {
  const char* name;
  uint64_t offset;  // in weights.bin
  uint64_t bytes;
};

inline constexpr Section kCpuSections[] = {
    {"TOKENIZER_MODEL", 128, 4'683'319},
    {"TF_LITE_VISION_ENCODER", 4'683'520, 150'460'932},
    {"TF_LITE_VISION_ADAPTER", 155'144'576, 17'847'812},
    {"TF_LITE_PREFILL_DECODE", 172'992'512, 1'408'555'140},
    {"TF_LITE_PER_LAYER_EMBEDDER", 1'581'547'776, 1'009'786'256},
    {"TF_LITE_EMBEDDER", 2'591'334'144, 271'583'748},
    {"METADATA", 2'862'918'016, 173},
};

// ---------------------------------------------------------------------------
// Tokenizer: identical in both variants [api]

inline constexpr uint32_t kVocabSize = 262'144;  // [verified] = Gemma 3n text vocab

// [verified] Every token the public Gemma 3n tokenizer defines is identical,
// and text is split into the same tokens, except two renamed turn markers.
// Of the slots Gemma 3n leaves as <unusedN>, 6,242 are named in Nano:
// <ctrl1>..<ctrl3360>, object detection (<start_of_box>, <start_of_polyline>,
// <x_bin_N>, <y_bin_N>, <rotation_N>, ...), video (<start_of_video>) and
// visual reasoning (<start_of_visual_reasoning>) tokens. Special tokens carry
// a 0xFF marker byte in the library's vocabulary (see nano::Tokenizer).
inline constexpr uint32_t kPad = 0;           // <pad>
inline constexpr uint32_t kEos = 1;           // <eos>
inline constexpr uint32_t kBos = 2;           // <bos>
inline constexpr uint32_t kUnk = 3;           // <unk>
inline constexpr uint32_t kMask = 4;          // <mask>
inline constexpr uint32_t kMultimodal = 5;    // [multimodal]
inline constexpr uint32_t kToxicity0 = 6;     // [toxicity=0]  (Gemma: <unused0>)
inline constexpr uint32_t kStartOfTurn = 105; // <ctrl99>   (Gemma: <start_of_turn>)
inline constexpr uint32_t kEndOfTurn = 106;   // <ctrl100>  (Gemma: <end_of_turn>)
inline constexpr uint32_t kStartOfImage = 255'999;  // [verified] boi_token_id
inline constexpr uint32_t kStartOfAudio = 256'000;  // [verified] boa_token_id
inline constexpr uint32_t kEosIds[] = {kEos, kEndOfTurn};  // [verified]

// ---------------------------------------------------------------------------
// Language model: Gemma 3n E2B decoder

inline constexpr uint32_t kHiddenSize = 2048;        // [verified] embedder size
inline constexpr uint32_t kNumLayers = 30;           // [verified] per-layer embedder
                                                     // size; rules out E4B (35)
inline constexpr uint32_t kPerLayerInputSize = 256;  // [verified] per-layer embedder
inline constexpr uint32_t kIntermediateSize = 8192;  // [public] MLP, every layer
inline constexpr uint32_t kNumAttentionHeads = 8;    // [public]
inline constexpr uint32_t kNumKeyValueHeads = 2;     // [public] grouped-query attention
inline constexpr uint32_t kHeadDim = 256;            // [public]
inline constexpr uint32_t kNumKvSharedLayers = 10;   // [public] last 10 layers reuse KV
inline constexpr uint32_t kSlidingWindow = 512;      // [public]
inline constexpr uint32_t kMaxPositions = 32'768;    // [public] (gnano uses 4,096)
inline constexpr uint32_t kAltUpInputs = 4;          // [public] AltUp
inline constexpr uint32_t kLaurelRank = 64;          // [public] LAuReL
inline constexpr float kRopeTheta = 1'000'000.0f;    // [public] global layers
inline constexpr float kRopeLocalTheta = 10'000.0f;  // [public] sliding layers
inline constexpr float kFinalLogitSoftcap = 30.0f;   // [public]
// [public] gelu_pytorch_tanh activation. Layer pattern: 4 sliding-window
// layers then 1 global layer, repeated (S S S S F) x 6.
inline constexpr char kLayerPattern[] = "SSSSFSSSSFSSSSFSSSSFSSSSFSSSSF";

// [measured] Both embedding tables are int4 with 12 bytes of per-row overhead
// (scales). TF_LITE_PREFILL_DECODE is 1,408,555,140 bytes for ~1.90B decoder
// weights implied by the config: ~5.9 bits per weight, so a 4/8-bit mix or
// extra tensors; the exact layout is in the encrypted header.
inline constexpr uint32_t kEmbeddingBits = 4;

// ---------------------------------------------------------------------------
// Other inputs

// [verified] Vision: MobileNet-V5-300M encoder (TF_LITE_VISION_ENCODER is
// 301M values at 4 bits) plus a projection into the language model. Both
// variants take images (gnano --image); each one occupies the prompt as
// <start_of_image>, 256 image tokens, <end_of_image>.
inline constexpr uint32_t kVisionTokensPerImage = 256;  // [verified]
inline constexpr uint32_t kTokensPerImage = kVisionTokensPerImage + 2;  // [verified]
// The images are passed as Skia SkBitmaps (see SkBitmap in engine.cc).
// [api] The GPU variant reports image and audio input. [measured] The CPU
// file has no audio section, so its audio support is unknown/absent.
inline constexpr uint32_t kAudioTokensPerClip = 188;  // [public] Gemma 3n audio

}  // namespace nano::model

#endif  // GEMINI_NANO_MODEL_H_
