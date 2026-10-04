// Engine: loads Chrome's liboptimization_guide_internal.so directly, asks it
// for the ChromeML C API table (GetChromeMLAPIV2), loads weights.bin and runs
// conversations -- the same calls Chrome's on-device model service makes (see
// chromium/ref/on_device_model_executor.cc, chromium/ref/session_accessor.cc).
//
// Must be compiled against Chromium's libc++ headers in ABI v2 / std::__Cr
// (third_party/libcxx), because the API passes std::function, std::variant,
// std::string etc. across the boundary. See the Makefile.

#include "engine.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Skia's SkBitmap, laid out as at Chrome 154's Skia revision (2466dcf3):
//   sk_sp<SkPixelRef> fPixelRef;
//   SkPixmap { const void* fPixels; size_t fRowBytes;
//              SkImageInfo { SkColorInfo { sk_sp<SkColorSpace>, SkColorType,
//                                          SkAlphaType }, SkISize } }
// The library copies and destroys these with its own Skia code; ours are
// built and freed by MakeBitmap() / ImageInput below.
class SkBitmap {
 public:
  void* pixel_ref;
  const void* pixels;
  size_t row_bytes;
  void* color_space;  // null: sRGB
  int32_t color_type;
  int32_t alpha_type;
  int32_t width;
  int32_t height;
};
static_assert(sizeof(SkBitmap) == 48, "SkBitmap layout mismatch");

#include "services/on_device_model/ml/chrome_ml_api.h"

// libc++ calls this on hardening failures; normally it lives in libc++.so.
namespace std {
inline namespace __Cr {
[[noreturn]] void __libcpp_verbose_abort(const char* fmt, ...) noexcept {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  abort();
}
}  // namespace __Cr
}  // namespace std

// libc++ normally ships these std::string members compiled into libc++.so;
// we have no such library in std::__Cr, so instantiate them here.
template class std::basic_string<char>;
template std::string& std::string::__assign_no_alias<true>(const char*, size_t);
template std::string& std::string::__assign_no_alias<false>(const char*, size_t);
template std::string std::operator+(const char*, const std::string&);

// The tool/audio structs declare out-of-line special members (defined in
// Chrome). Our variant instantiation needs them, so define them here.
namespace ml {
AudioBuffer::AudioBuffer() = default;
AudioBuffer::~AudioBuffer() = default;
AudioBuffer::AudioBuffer(const AudioBuffer&) = default;
AudioBuffer& AudioBuffer::operator=(const AudioBuffer&) = default;
AudioBuffer::AudioBuffer(AudioBuffer&&) = default;
AudioBuffer& AudioBuffer::operator=(AudioBuffer&&) = default;
#define ML_DEFAULTS(T)                       \
  T::T() = default;                          \
  T::~T() = default;                         \
  T::T(const T&) = default;                  \
  T& T::operator=(const T&) = default;       \
  T::T(T&&) = default;                       \
  T& T::operator=(T&&) = default;
ML_DEFAULTS(ToolCall)
ML_DEFAULTS(ToolResponse)
ML_DEFAULTS(ToolDeclaration)
}  // namespace ml

static_assert(sizeof(ml::InputPiece) == 104, "InputPiece layout mismatch");

namespace nano {
namespace {

// ---------------------------------------------------------------------------
// A tiny thread pool for ChromeMLScheduleFn: the library hands us tasks and
// expects them to run on some other thread.

struct Pool {
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
  std::vector<std::function<void()>*> queue;
};
Pool g_pool;

void* Worker(void*) {
  for (;;) {
    pthread_mutex_lock(&g_pool.mu);
    while (g_pool.queue.empty()) pthread_cond_wait(&g_pool.cv, &g_pool.mu);
    std::function<void()>* task = g_pool.queue.front();
    g_pool.queue.erase(g_pool.queue.begin());
    pthread_mutex_unlock(&g_pool.mu);
    (*task)();
    delete task;
  }
  return nullptr;
}

void StartWorkers() {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, [] {
    for (int i = 0; i < 4; ++i) {
      pthread_t t;
      pthread_create(&t, nullptr, Worker, nullptr);
      pthread_detach(t);
    }
  });
}

void Schedule(uintptr_t, std::function<void()>* task) {
  auto* owned = new std::function<void()>(std::move(*task));
  pthread_mutex_lock(&g_pool.mu);
  g_pool.queue.push_back(owned);
  pthread_cond_signal(&g_pool.cv);
  pthread_mutex_unlock(&g_pool.mu);
}

// One-shot latch for waiting on library callbacks.
struct Latch {
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
  bool done = false;
  void Signal() {
    pthread_mutex_lock(&mu);
    done = true;
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mu);
  }
  void Wait() {
    pthread_mutex_lock(&mu);
    while (!done) pthread_cond_wait(&cv, &mu);
    pthread_mutex_unlock(&mu);
  }
};

// Per-user data: the models (profile-gpu/, profile-cpu/) and the copy of
// Google's library (lib/), as saved by download-model.py. $NANO_HOME, else
// $XDG_DATA_HOME/gemini-nano, else ~/.local/share/gemini-nano.
std::string DataDir() {
  if (const char* env = getenv("NANO_HOME")) return env;
  if (const char* xdg = getenv("XDG_DATA_HOME")) return std::string(xdg) + "/gemini-nano";
  const char* home = getenv("HOME");
  return std::string(home ? home : ".") + "/.local/share/gemini-nano";
}

// Folder holding the real executable (symlinks in bin/ resolved): our Dawn
// build, libnano_dawn.so, is installed next to it.
std::string ExeDir() {
  char buf[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return ".";
  std::string path(buf, n);
  return path.substr(0, path.rfind('/'));
}

// Newest model folder under <profile>/OptGuideOnDeviceModel/, or "".
// Version folders sort in increasing order, so the last match is newest.
std::string FindModel(const std::string& profile) {
  std::string pattern = profile + "/OptGuideOnDeviceModel/*/weights.bin";
  std::string dir;
  glob_t g;
  if (glob(pattern.c_str(), 0, nullptr, &g) == 0) {
    dir = g.gl_pathv[g.gl_pathc - 1];
    dir.resize(dir.size() - strlen("/weights.bin"));
  }
  globfree(&g);
  return dir;
}

double Now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

[[noreturn]] void Fatal(const char* msg) {
  fprintf(stderr, "\n[chromeml fatal] %s\n", msg);
  abort();
}
void NoopExactLinear(const char*, int, int) {}
void NoopCustomCounts(const char*, int, int, int, size_t) {}
void NoopMediumTimes(const char*, int64_t) {}

// Chrome calls TryInitDawnProcs() with its Dawn (WebGPU) function table before
// anything else, and the library initializes its CPU-topology code (cpuinfo)
// there; skipping it aborts model loading. The library checks that the table
// starts with the 20-byte Dawn git revision it was built with (Chromium 154
// DEPS 'dawn_revision') and copies sizeof(DawnProcTable) = 0x8b8 bytes.
//
// For the GPU backend we pass the real table from libnano_dawn.so (Dawn built
// at that same revision, see src/dawn/). For the CPU backend, a zeroed table
// with the right header is enough: it never calls into it.
constexpr unsigned char kDawnRevision[20] = {
    0x85, 0x97, 0xa1, 0xaa, 0xec, 0x54, 0x6a, 0x7f, 0x19, 0xe8,
    0xe6, 0x62, 0xa4, 0x9e, 0xbf, 0x03, 0xb2, 0xe9, 0x13, 0x08};
constexpr size_t kDawnProcTableSize = 0x8b8;
alignas(8) unsigned char g_dawn_procs[kDawnProcTableSize];

// ---------------------------------------------------------------------------
// Image input. The library takes images as SkBitmaps, which hold their pixels
// through a reference-counted SkPixelRef. We build one byte-for-byte as at
// Chrome 154's Skia revision and keep our own reference on it, so the
// library's copies never drop the count to zero and never call its virtual
// destructor. The vtable's entries abort loudly if the library calls them
// anyway.

constexpr int32_t kBGRA_8888_SkColorType = 6;  // kN32 on Linux
constexpr int32_t kPremul_SkAlphaType = 2;
constexpr int32_t kSkPixelStorageType_PixelRef = 1;

// SkPixelRef field offsets (vtable at 0):
//   8 fRefCnt, 12 SkPixelStorage::fType, 16 fID, 20 fContentID,
//   24 fWidth, 28 fHeight, 32 fPixels, 40 fRowBytes, 48 fTaggedGenID,
//   56 SkIDChangeListener::List { SkMutex (SkSemaphore: 56 fCount = 1,
//      60 SkOnce, 64 OSSemaphore*), STArray<1, sk_sp<...>> (72 fData,
//      80 fSize, 84 fOwnMemory:1 + fCapacity:31, 88 inline storage) },
//   96 fAddedToCache, 97 fMutability : 8.
constexpr size_t kPixelRefSize = 104;

[[noreturn]] void UnexpectedPixelRefCall(const void*) {
  Fatal("the model library called a virtual method on our SkPixelRef");
}
void* NoDiscardable(const void*) { return nullptr; }

// Itanium C++ ABI vtable: offset-to-top, RTTI, then the virtual functions of
// SkRefCntBase (complete dtor, deleting dtor, internal_dispose) and SkPixelRef
// (diagnostic_only_getDiscardable).
const void* const kPixelRefVtable[] = {
    nullptr, nullptr,
    reinterpret_cast<const void*>(&UnexpectedPixelRefCall),
    reinterpret_cast<const void*>(&UnexpectedPixelRefCall),
    reinterpret_cast<const void*>(&UnexpectedPixelRefCall),
    reinterpret_cast<const void*>(&NoDiscardable),
};

template <typename T>
void Put(unsigned char* base, size_t offset, T value) {
  memcpy(base + offset, &value, sizeof(T));
}

// Owns the pixel refs of one Generate() call.
class ImageInput {
 public:
  ImageInput() = default;
  ImageInput(const ImageInput&) = delete;
  ImageInput& operator=(const ImageInput&) = delete;

  ~ImageInput() {
    for (unsigned char* ref : refs_) {
      int32_t count;
      memcpy(&count, ref + 8, sizeof(count));
      // Only ours left: free it. Otherwise the library still holds a copy.
      if (count == 1) delete[] ref;
    }
  }

  SkBitmap MakeBitmap(const Image& image) {
    auto* ref = new unsigned char[kPixelRefSize]();  // 16-byte aligned
    refs_.push_back(ref);
    void* pixels = const_cast<uint8_t*>(image.bgra.data());
    size_t row_bytes = (size_t)image.width * 4;
    static uint32_t next_id = 1;
    Put<const void*>(ref, 0, &kPixelRefVtable[2]);
    Put<int32_t>(ref, 8, 1);  // our reference
    Put<int32_t>(ref, 12, kSkPixelStorageType_PixelRef);
    Put<uint32_t>(ref, 16, next_id);
    Put<int32_t>(ref, 24, image.width);
    Put<int32_t>(ref, 28, image.height);
    Put<void*>(ref, 32, pixels);
    Put<size_t>(ref, 40, row_bytes);
    Put<uint32_t>(ref, 48, 2 * next_id++);  // even: not a unique-tagged id
    Put<int32_t>(ref, 56, 1);               // SkMutex starts unlocked
    Put<void*>(ref, 72, ref + 88);          // STArray uses inline storage
    Put<uint32_t>(ref, 84, 1u << 1);        // capacity 1, not owned
    return SkBitmap{ref, pixels, row_bytes, nullptr, kBGRA_8888_SkColorType,
                    kPremul_SkAlphaType, image.width, image.height};
  }

 private:
  std::vector<unsigned char*> refs_;
};

// Matches Chrome's model max_tokens; prompts plus replies must fit in it.
constexpr uint32_t kMaxTokens = 4096;

}  // namespace

bool ResolveLoadOptions(LoadOptions* opts, std::string* err) {
  auto env = [](const char* name) {
    const char* v = getenv(name);
    return std::string(v ? v : "");
  };
  const std::string data = DataDir();
  // download-model.py stores each model variant in its own profile folder.
  if (opts->backend.empty()) opts->backend = env("NANO_BACKEND");
  if (opts->backend.empty())
    opts->backend = FindModel(data + "/profile-gpu").empty() ? "cpu" : "gpu";
  if (opts->backend != "gpu" && opts->backend != "cpu") {
    *err = "NANO_BACKEND must be gpu or cpu";
    return false;
  }
  if (opts->model_dir.empty()) opts->model_dir = env("NANO_MODEL_DIR");
  if (opts->model_dir.empty()) {
    const std::string profile = data + "/profile-" + opts->backend;
    opts->model_dir = FindModel(profile);
    if (opts->model_dir.empty()) {
      *err = "No " + opts->backend + " model found in " + profile +
             ". Run gnano-download --" + opts->backend +
             " first, or set NANO_MODEL_DIR.";
      return false;
    }
  }
  if (opts->lib_path.empty()) opts->lib_path = env("NANO_LIB");
  if (opts->lib_path.empty()) {
    opts->lib_path = data + "/lib/liboptimization_guide_internal.so";
    if (access(opts->lib_path.c_str(), R_OK) != 0)
      opts->lib_path = "/opt/google/chrome/liboptimization_guide_internal.so";
  }
  if (opts->dawn_path.empty()) opts->dawn_path = env("NANO_DAWN_LIB");
  if (opts->dawn_path.empty()) opts->dawn_path = ExeDir() + "/libnano_dawn.so";
  return true;
}

Engine::~Engine() {
  if (model_) api_->DestroyModel(model_);
}

bool Engine::Load(const LoadOptions& opts, std::string* err) {
  opts_ = opts;
  bool gpu = opts.backend == "gpu";
  StartWorkers();

  // 1. Load the library and get the API table.
  void* lib = dlopen(opts.lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    *err = std::string("dlopen failed: ") + dlerror();
    return false;
  }
  auto get_api =
      reinterpret_cast<ChromeMLAPIGetterV2>(dlsym(lib, "GetChromeMLAPIV2"));
  if (!get_api) {
    *err = "GetChromeMLAPIV2 not found in " + opts.lib_path;
    return false;
  }
  api_ = get_api(ChromeMLBackendMode::kLiteRtLmSession);
  if (!api_) {
    *err = "GetChromeMLAPIV2 returned null";
    return false;
  }
  fprintf(stderr, "[ok] loaded %s, got ChromeMLAPI table\n",
          opts.lib_path.c_str());

  static const ChromeMLMetricsFns metrics = {NoopExactLinear, NoopCustomCounts,
                                             NoopMediumTimes};
  api_->SetMetricsFns(&metrics);
  api_->SetFatalErrorFn(Fatal);
  api_->SetFatalErrorNonGpuFn(Fatal);

  const DawnProcTable* procs = nullptr;
  if (gpu) {
    void* dawn = dlopen(opts.dawn_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!dawn) {
      *err = std::string("dlopen failed: ") + dlerror() + "\nRun make first.";
      return false;
    }
    auto get_procs = reinterpret_cast<const DawnProcTable* (*)()>(
        dlsym(dawn, "NanoGetDawnProcs"));
    auto get_size =
        reinterpret_cast<unsigned (*)()>(dlsym(dawn, "NanoDawnProcTableSize"));
    if (!get_procs || !get_size || get_size() != kDawnProcTableSize) {
      *err = opts.dawn_path + " has the wrong Dawn version";
      return false;
    }
    procs = get_procs();
  } else {
    memcpy(g_dawn_procs, kDawnRevision, sizeof(kDawnRevision));
    procs = reinterpret_cast<const DawnProcTable*>(g_dawn_procs);
  }
  if (!api_->TryInitDawnProcs(*procs)) {
    *err =
        "TryInitDawnProcs failed: this library is not from Chrome 154 (Dawn "
        "revision mismatch).";
    return false;
  }

  // 2. Load the model.
  std::string weights = opts.model_dir + "/weights.bin";
  int fd = open(weights.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    *err = weights + ": " + strerror(errno);
    return false;
  }
  ChromeMLModelData data;
  data.weights_file = fd;  // the library takes ownership
  // Like Chrome, give the CPU backend the pre-packed weight cache that comes
  // with the CPU model (cache.bin), so it can skip repacking the weights.
  if (!gpu) {
    std::string cache = opts.model_dir + "/cache.bin";
    data.cache_file = open(cache.c_str(), O_RDONLY | O_CLOEXEC);
  }

  ChromeMLModelDescriptor desc = {
      .backend_type = gpu ? ml::ModelBackendType::kGpuBackend
                          : ml::ModelBackendType::kCpuBackend,
      .model_data = &data,
      .max_tokens = kMaxTokens,
      .temperature = 0.0f,
      .top_k = 128,
      .adaptation_ranks = nullptr,
      .adaptation_ranks_size = 0,
      // Chrome's defaults (kPreferTextureWeights, kEnableHostMappedPointer).
      .prefer_texture_weights = gpu,
      .enable_host_mapped_pointer = gpu,
      .use_low_power = false,
      .allow_fp16 = true,
      .enable_speculative_decoding = false,
      .performance_hint = ml::ModelPerformanceHint::kHighestQuality,
      .vram_mb = 0,
  };
  fprintf(stderr, "[..] loading %s (%s backend)...\n", weights.c_str(),
          gpu ? "GPU" : "CPU");
  double t0 = Now();
  model_ = api_->SessionCreateModel(&desc, 0, Schedule);
  if (!model_) {
    *err = "SessionCreateModel failed (see the log above)";
    return false;
  }
  fprintf(stderr, "[ok] model loaded in %.1fs\n", Now() - t0);
  return true;
}

bool Engine::Generate(const std::vector<Message>& messages,
                      const GenerateParams& params,
                      const TextFn& on_text,
                      GenerateStats* stats,
                      std::string* err) {
  ChromeMLAdaptationDescriptor sdesc = {
      .model_data = nullptr,
      .max_tokens = 0,
      .top_k = params.top_k < ml::kMinTopK ? ml::kMinTopK : params.top_k,
      .temperature = params.temperature < ml::kMinTemperature
                         ? ml::kMinTemperature
                         : params.temperature,
      .enable_image_input = false,
      .enable_audio_input = false,
  };
  for (const Message& m : messages)
    if (!m.images.empty()) sdesc.enable_image_input = true;
  ChromeMLSession session = api_->CreateSession(model_, &sdesc);
  if (!session) {
    *err = "CreateSession failed";
    return false;
  }
  ChromeMLCancel cancel = api_->CreateCancel();
  bool ok = false;

  // "<role>text<end>" per message, then "<model>" to start the reply. The
  // control tokens are emitted by the library from ml::Token values.
  // Declared before `input`, so the pixels outlive every piece using them.
  ImageInput images;
  std::vector<ml::InputPiece> input;
  for (const Message& m : messages) {
    input.emplace_back(m.role == Role::kSystem ? ml::Token::kSystem
                       : m.role == Role::kUser ? ml::Token::kUser
                                               : ml::Token::kModel);
    for (const Image& image : m.images) input.emplace_back(images.MakeBitmap(image));
    input.emplace_back(m.content);
    input.emplace_back(ml::Token::kEnd);
  }
  input.emplace_back(ml::Token::kModel);

  Latch appended;
  ChromeMLContextSavedFn saved_fn = [&](int n) {
    stats->prompt_tokens = n;
    appended.Signal();
  };
  ChromeMLAppendOptions aopts = {
      .input = input.data(),
      .input_size = input.size(),
      .max_tokens = kMaxTokens,
      .context_saved_fn = &saved_fn,
      .input_source = InputSource::kUserInput,
  };
  double t0 = Now();
  if (!api_->SessionAppend(session, &aopts, cancel)) {
    *err = "SessionAppend failed";
  } else {
    appended.Wait();
    stats->prefill_seconds = Now() - t0;

    Latch finished;
    int chunks = 0;
    bool stopped = false;
    // The model often ends its reply with blank lines. Hold whitespace back
    // until more text follows, so trailing whitespace is never emitted.
    std::string held;
    ChromeMLGenerateOutputFn out_fn = [&](const ChromeMLGenerateOutput* out) {
      if (out->text && !stopped) {
        ++chunks;
        std::string text = held + out->text;
        size_t keep = text.find_last_not_of(" \t\r\n");
        size_t cut = keep == std::string::npos ? 0 : keep + 1;
        held = text.substr(cut);
        text.resize(cut);
        if (!text.empty() && !on_text(text.c_str())) {
          stopped = true;
          api_->CancelExecuteModel(cancel);
        }
      }
      if (out->status != ChromeMLGenerateStatus::kInProgress) {
        stats->output_tokens =
            out->tokens_decoded >= 0 ? out->tokens_decoded : chunks;
        finished.Signal();
      }
    };
    ChromeMLGenerateOptions gopts = {
        .max_output_tokens = params.max_output_tokens,
        .constraint = 0,
        .output_fn = &out_fn,
    };
    t0 = Now();
    if (!api_->SessionGenerate(session, &gopts, cancel)) {
      *err = "SessionGenerate failed";
    } else {
      finished.Wait();
      stats->decode_seconds = Now() - t0;
      ok = true;
    }
  }

  api_->DestroyCancel(cancel);
  api_->DestroySession(session);
  return ok;
}

bool Engine::GetTokenizer(Tokenizer* out, std::string* err) {
  // Chrome asks for this to build its constrained-decoding tokenizer. It needs
  // a session; the callback runs before GetTokenizerParamsV3 returns or on a
  // library thread, so wait for it either way.
  ChromeMLAdaptationDescriptor sdesc = {};
  ChromeMLSession session = api_->CreateSession(model_, &sdesc);
  if (!session) {
    *err = "CreateSession failed";
    return false;
  }
  Latch done;
  ChromeMLGetTokenizerParamsV3Fn fn = [&](const ChromeMLTokenizerParamsV3& p) {
    out->eos_ids.assign(p.eos_token_ids, p.eos_token_ids + p.eos_token_ids_size);
    if (p.tokenizer_json_file_content)
      out->tokenizer_json = p.tokenizer_json_file_content;
    if (p.token_lens && p.token_bytes) {
      out->vocab.reserve(p.vocab_size);
      const uint8_t* b = p.token_bytes;
      for (uint32_t i = 0; i < p.vocab_size; ++i) {
        out->vocab.emplace_back(reinterpret_cast<const char*>(b), p.token_lens[i]);
        b += p.token_lens[i];
      }
    }
    out->tokenize_fn = p.tokenize_fn;
    out->tokenize_user_data = p.tokenize_user_data;
    done.Signal();
  };
  bool ok = api_->GetTokenizerParamsV3(model_, session, fn);
  if (ok) done.Wait();
  api_->DestroySession(session);
  if (!ok) *err = "GetTokenizerParamsV3 failed";
  return ok;
}

bool Engine::GetCapabilities(Capabilities* out, std::string* err) {
  std::string weights = opts_.model_dir + "/weights.bin";
  int fd = open(weights.c_str(), O_RDONLY | O_CLOEXEC);  // the library owns it
  if (fd < 0) {
    *err = weights + ": " + strerror(errno);
    return false;
  }
  ChromeMLCapabilities caps;
  if (!api_->GetCapabilities(fd, caps)) {
    *err = "GetCapabilities failed";
    return false;
  }
  out->image_input = caps.image_input;
  out->audio_input = caps.audio_input;
  return true;
}

std::vector<uint32_t> Tokenizer::Encode(const std::string& text) const {
  std::vector<uint32_t> ids(text.size() + 16);
  auto bytes = reinterpret_cast<const uint8_t*>(text.data());
  size_t n = tokenize_fn(tokenize_user_data, bytes, text.size(), ids.data(), ids.size());
  if (n > ids.size()) {
    ids.resize(n);
    n = tokenize_fn(tokenize_user_data, bytes, text.size(), ids.data(), ids.size());
  }
  ids.resize(n);
  return ids;
}

}  // namespace nano
