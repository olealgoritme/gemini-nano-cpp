// gnano: run Chrome's on-device Gemini Nano without Chrome, from the
// command line. All the model work lives in engine.cc.
//
//   gnano what is a haiku           (quotes optional)
//   echo "summarize this" | gnano
//   gnano -v "hi"                   (show library logs and speed)
//   gnano --image cat.jpg what is this   (image input, repeatable)
//   gnano --info                    (what the library reports, as JSON)
//   gnano --dump-tokenizer vocab.json
//   gnano --tokenize hello world    (token ids, as JSON)

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "engine.h"
#include "json.h"

namespace {

void Usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s [-v] [--image FILE]... <prompt...>\n"
          "       echo <prompt> | %s [-v]\n"
          "       %s --info\n"
          "       %s --dump-tokenizer <file.json>\n"
          "       %s --tokenize <text...>\n"
          "\n"
          "  -v                show the model library's logs and generation speed\n"
          "  --image FILE      show the model an image (JPEG, PNG, GIF, BMP);\n"
          "                    repeat for several. Without a prompt: describe it\n"
          "  --info            print what the library reports about the model\n"
          "                    (tokenizer, special tokens, capabilities) as JSON\n"
          "  --dump-tokenizer  write the full vocabulary to a JSON file\n"
          "  --tokenize        print the token ids of the text as JSON\n"
          "\n"
          "env:\n"
          "  NANO_BACKEND    gpu or cpu (default: gpu if a GPU model is\n"
          "                  downloaded, else cpu)\n"
          "  NANO_HOME       folder holding the models and lib/ (default:\n"
          "                  ~/.local/share/gemini-nano)\n"
          "  NANO_CTX        context window in tokens (default: 4096)\n"
          "  NANO_MODEL_DIR  folder holding weights.bin\n"
          "  NANO_LIB        liboptimization_guide_internal.so\n"
          "  NANO_DAWN_LIB   our Dawn build (libnano_dawn.so)\n",
          argv0, argv0, argv0, argv0, argv0);
}

// Special tokens carry a 0xFF marker byte (see nano::Tokenizer). The token
// that is just the raw byte 0xFF is not marked.
bool IsSpecial(const std::string& token) {
  return token.size() > 1 && (unsigned char)token[0] == 0xFF;
}
std::string TokenText(const std::string& token) {
  return IsSpecial(token) ? token.substr(1) : token;
}

bool IsUtf8(const std::string& s) {
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (!n || i + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k)
      if (((unsigned char)s[i + k] >> 6) != 2) return false;
    i += n;
  }
  return true;
}

// A token as JSON: a string, or {"hex": "..."} for bytes that aren't UTF-8.
std::string TokenJson(const std::string& text) {
  if (IsUtf8(text)) return json::Quote(text);
  std::string hex;
  for (unsigned char c : text) {
    char b[3];
    snprintf(b, sizeof(b), "%02x", c);
    hex += b;
  }
  return "{\"hex\":\"" + hex + "\"}";
}

int PrintInfo(nano::Engine& engine, const nano::LoadOptions& opts) {
  std::string err;
  nano::Tokenizer tok;
  if (!engine.GetTokenizer(&tok, &err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  // The library only answers this for some model files (the GPU one); the
  // CPU model's container is rejected, so report null there.
  nano::Engine::Capabilities caps;
  bool have_caps = engine.GetCapabilities(&caps, &err);
  std::string out = "{\n  \"backend\": " + json::Quote(opts.backend) +
                    ",\n  \"model_dir\": " + json::Quote(opts.model_dir) +
                    ",\n  \"capabilities\": " +
                    (!have_caps ? std::string("null")
                                : std::string("{\"image_input\": ") +
                                      (caps.image_input ? "true" : "false") +
                                      ", \"audio_input\": " +
                                      (caps.audio_input ? "true" : "false") + "}") +
                    ",\n  \"tokenizer\": {\n    \"vocab_size\": ";
  char num[32];
  snprintf(num, sizeof(num), "%zu", tok.vocab.size());
  out += num;
  out += ",\n    \"eos_ids\": [";
  for (size_t i = 0; i < tok.eos_ids.size(); ++i) {
    snprintf(num, sizeof(num), "%s%u", i ? ", " : "", tok.eos_ids[i]);
    out += num;
  }
  out += "],\n    \"eos_tokens\": {";
  bool first = true;
  for (size_t id = 0; id < tok.vocab.size(); ++id) {
    if (!IsSpecial(tok.vocab[id])) continue;
    snprintf(num, sizeof(num), "%s\n      \"%zu\": ", first ? "" : ",", id);
    out += num + TokenJson(TokenText(tok.vocab[id]));
    first = false;
  }
  out += "\n    }\n  }\n}\n";
  fputs(out.c_str(), stdout);
  return 0;
}

int Tokenize(nano::Engine& engine, const std::string& text) {
  std::string err;
  nano::Tokenizer tok;
  if (!engine.GetTokenizer(&tok, &err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  std::vector<uint32_t> ids = tok.Encode(text);
  printf("[");
  for (size_t i = 0; i < ids.size(); ++i) printf("%s%u", i ? ", " : "", ids[i]);
  printf("]\n");
  return 0;
}

int DumpTokenizer(nano::Engine& engine, const char* path) {
  std::string err;
  nano::Tokenizer tok;
  if (!engine.GetTokenizer(&tok, &err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  FILE* f = fopen(path, "w");
  if (!f) {
    perror(path);
    return 1;
  }
  // {"vocab": [token text by id], "special_ids": [...], "eos_ids": [...]}
  std::string out = "{\"vocab\": [";
  std::string special;
  char num[32];
  for (size_t id = 0; id < tok.vocab.size(); ++id) {
    if (id) out += ",";
    out += "\n" + TokenJson(TokenText(tok.vocab[id]));
    if (IsSpecial(tok.vocab[id])) {
      snprintf(num, sizeof(num), "%s%zu", special.empty() ? "" : ", ", id);
      special += num;
    }
  }
  out += "\n], \"special_ids\": [" + special + "], \"eos_ids\": [";
  for (size_t i = 0; i < tok.eos_ids.size(); ++i) {
    snprintf(num, sizeof(num), "%s%u", i ? ", " : "", tok.eos_ids[i]);
    out += num;
  }
  out += "]}\n";
  fwrite(out.data(), 1, out.size(), f);
  fclose(f);
  fprintf(stderr, "Wrote %zu tokens to %s\n", tok.vocab.size(), path);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = false, info = false, tokenize = false;
  const char* dump_path = nullptr;
  std::vector<std::string> image_paths;
  std::string prompt;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) {
      verbose = true;
    } else if (!strcmp(argv[i], "--info")) {
      info = true;
    } else if (!strcmp(argv[i], "--tokenize")) {
      tokenize = true;
    } else if (!strcmp(argv[i], "--dump-tokenizer") && i + 1 < argc) {
      dump_path = argv[++i];
    } else if (!strcmp(argv[i], "--image") && i + 1 < argc) {
      image_paths.push_back(argv[++i]);
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      Usage(argv[0]);
      return 0;
    } else {
      if (!prompt.empty()) prompt += ' ';
      prompt += argv[i];
    }
  }
  bool query = info || dump_path;
  if ((tokenize || !query) && prompt.empty() && !isatty(STDIN_FILENO)) {
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0) prompt.append(buf, n);
  }
  if (prompt.empty() && !image_paths.empty() && !query && !tokenize)
    prompt = "Describe this image.";
  if ((tokenize || !query) && prompt.empty()) {
    Usage(argv[0]);
    return 2;
  }

  // Decode images before loading the model, to fail fast on a bad file.
  std::vector<nano::Image> images;
  for (const std::string& path : image_paths) {
    std::string bytes, err;
    images.emplace_back();
    if (!nano::ReadFile(path, &bytes, &err) ||
        !nano::DecodeImage(bytes, &images.back(), &err)) {
      fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
  }

  // The library logs a lot to stderr; hide it unless -v.
  int saved_stderr = -1;
  if (!verbose) {
    saved_stderr = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDERR_FILENO);
    close(devnull);
  }
  auto fail = [&](const std::string& err) {
    if (saved_stderr >= 0) dup2(saved_stderr, STDERR_FILENO);
    fprintf(stderr, "\n%s\n", err.c_str());
    if (!verbose) fprintf(stderr, "(run with -v to see the library's logs)\n");
    return 1;
  };

  nano::LoadOptions opts;
  nano::Engine engine;
  std::string err;
  if (!nano::ResolveLoadOptions(&opts, &err) || !engine.Load(opts, &err))
    return fail(err);
  if (query || tokenize) {
    if (saved_stderr >= 0) dup2(saved_stderr, STDERR_FILENO);
    if (tokenize) return Tokenize(engine, prompt);
    return info ? PrintInfo(engine, opts) : DumpTokenizer(engine, dump_path);
  }

  std::vector<nano::Message> messages = {{nano::Role::kUser, prompt, std::move(images)}};
  nano::GenerateStats stats;
  bool ok = engine.Generate(
      messages, nano::GenerateParams(),
      [](const char* text) {
        fputs(text, stdout);
        fflush(stdout);
        return true;
      },
      &stats, &err);
  if (!ok) return fail(err);
  fputc('\n', stdout);
  fflush(stdout);
  if (verbose) {
    fprintf(stderr,
            "[ok] prompt %d tokens in %.2fs, reply %d tokens in %.2fs "
            "(%.1f tok/s)\n",
            stats.prompt_tokens, stats.prefill_seconds, stats.output_tokens,
            stats.decode_seconds, stats.output_tokens / stats.decode_seconds);
  }
  return 0;
}
