// Engine: Gemini Nano through Chrome's liboptimization_guide_internal.so.
//
// Shared by gnano (command line) and gnano-server (OpenAI-compatible HTTP).
// Must be compiled against Chromium's libc++ like the rest (see the Makefile).

#ifndef ENGINE_H_
#define ENGINE_H_

#include <stdint.h>

#include <functional>
#include <string>
#include <vector>

#include "image.h"

struct ChromeMLAPI;

namespace nano {

enum class Role { kSystem, kUser, kAssistant };

struct Message {
  Role role;
  std::string content;
  std::vector<Image> images;  // shown to the model before `content`
};

struct LoadOptions {
  std::string backend;    // "gpu" or "cpu"
  std::string model_dir;  // folder holding weights.bin
  std::string lib_path;   // liboptimization_guide_internal.so
  std::string dawn_path;  // libnano_dawn.so (GPU only)
  uint32_t context_tokens = 0;  // prompt + reply budget; 0 = NANO_CTX or 4096
};

// Fills unset LoadOptions from NANO_* environment variables, the per-user data
// folder (~/.local/share/gemini-nano: profile-gpu/, profile-cpu/, lib/) and
// the program's own folder (libnano_dawn.so). Returns false with a
// user-facing message in `err` if no model can be found.
bool ResolveLoadOptions(LoadOptions* opts, std::string* err);

struct GenerateParams {
  // The Prompt API's defaults.
  uint32_t top_k = 3;
  float temperature = 1.0f;
  uint32_t max_output_tokens = 1024;
};

struct GenerateStats {
  int prompt_tokens = 0;
  int output_tokens = 0;
  double prefill_seconds = 0;
  double decode_seconds = 0;
};

// The model's tokenizer, as reported by the library (GetTokenizerParamsV3).
struct Tokenizer {
  // Token id -> token bytes. Special tokens (<eos>, <ctrl100>, ...) start with
  // a 0xFF marker byte, the convention of llguidance, which Chrome uses for
  // constrained decoding; it is not part of the token's text.
  std::vector<std::string> vocab;
  std::vector<uint32_t> eos_ids;    // end-of-sequence token ids
  std::string tokenizer_json;       // set instead of vocab by some models
  // Text -> token ids, through the library. Valid while the Engine lives.
  size_t (*tokenize_fn)(const void* user_data, const uint8_t* bytes,
                        size_t bytes_len, uint32_t* out, size_t out_len) = nullptr;
  const void* tokenize_user_data = nullptr;

  std::vector<uint32_t> Encode(const std::string& text) const;
};

// Receives each chunk of generated text. Return false to stop generating.
using TextFn = std::function<bool(const char* text)>;

class Engine {
 public:
  Engine() = default;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  // Destroys the model, which also stops the library's background work (such
  // as GPU shader compilation) before the process exits.
  ~Engine();

  // Loads the library and the model, logging progress to stderr.
  bool Load(const LoadOptions& opts, std::string* err);

  // Runs one conversation in a fresh session and streams the reply to
  // `on_text`. Not thread-safe: callers must serialize calls.
  bool Generate(const std::vector<Message>& messages,
                const GenerateParams& params,
                const TextFn& on_text,
                GenerateStats* stats,
                std::string* err);

  bool GetTokenizer(Tokenizer* out, std::string* err);

  // How many tokens `messages` take in the context window: exact for text, an
  // estimate for images and the role markers. Returns -1 (with `err`) if the
  // tokenizer is unavailable. Not thread-safe: callers must serialize calls.
  int CountPromptTokens(const std::vector<Message>& messages, std::string* err);

  // What the library says the model file supports (GetCapabilities).
  struct Capabilities {
    bool image_input = false;
    bool audio_input = false;
  };
  bool GetCapabilities(Capabilities* out, std::string* err);

  const LoadOptions& options() const { return opts_; }

 private:
  LoadOptions opts_;
  const ChromeMLAPI* api_ = nullptr;
  uintptr_t model_ = 0;
  Tokenizer tokenizer_;  // filled by the first CountPromptTokens
  bool have_tokenizer_ = false;
};

}  // namespace nano

#endif  // ENGINE_H_
