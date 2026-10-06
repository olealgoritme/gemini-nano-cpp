// gnano-server: an OpenAI-compatible HTTP API for Gemini Nano, without Chrome.
//
//   GET  /v1/models
//   POST /v1/chat/completions   (stream true/false, temperature, top_k,
//                                max_tokens / max_completion_tokens, images
//                                as base64 data: URIs in image_url parts)
//   GET  /healthz
//
// The model stays loaded; requests are handled one generation at a time.
// Plain sockets and a tiny JSON parser, so there is nothing to install.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "engine.h"
#include "json.h"

namespace {

constexpr char kModelName[] = "gemini-nano";
constexpr size_t kMaxRequestBytes = 64 << 20;  // room for base64 images

nano::Engine g_engine;
pthread_mutex_t g_engine_mu = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t g_id_mu = PTHREAD_MUTEX_INITIALIZER;
unsigned long g_next_id = 0;

std::string Fmt(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string Fmt(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return buf;
}

void Log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void Log(const char* fmt, ...) {
  time_t now = time(nullptr);
  char ts[16];
  strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&now));
  fprintf(stderr, "%s ", ts);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
}

bool WriteAll(int fd, const std::string& data) {
  size_t off = 0;
  while (off < data.size()) {
    ssize_t n = write(fd, data.data() + off, data.size() - off);
    if (n <= 0) return false;
    off += n;
  }
  return true;
}

const char* StatusText(int status) {
  switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 413: return "Payload Too Large";
    default: return "Internal Server Error";
  }
}

constexpr char kCors[] =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Headers: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

void SendJson(int fd, int status, const std::string& body) {
  WriteAll(fd, Fmt("HTTP/1.1 %d %s\r\n", status, StatusText(status)) +
                   "Content-Type: application/json\r\n" + kCors +
                   Fmt("Content-Length: %zu\r\n", body.size()) +
                   "Connection: close\r\n\r\n" + body);
}

void SendError(int fd, int status, const std::string& message,
               const char* code = nullptr) {
  SendJson(fd, status,
           "{\"error\":{\"message\":" + json::Quote(message) +
               ",\"type\":\"invalid_request_error\"" +
               (code ? std::string(",\"code\":\"") + code + "\"" : "") + "}}");
}

struct Request {
  std::string method, path, body;
};

// Reads one HTTP request (headers + Content-Length body).
bool ReadRequest(int fd, Request* req) {
  std::string data;
  char buf[8192];
  size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n <= 0 || data.size() > kMaxRequestBytes) return false;
    data.append(buf, n);
    header_end = data.find("\r\n\r\n");
  }
  size_t line_end = data.find("\r\n");
  std::string line = data.substr(0, line_end);
  size_t sp1 = line.find(' '), sp2 = line.rfind(' ');
  if (sp1 == std::string::npos || sp2 == sp1) return false;
  req->method = line.substr(0, sp1);
  req->path = line.substr(sp1 + 1, sp2 - sp1 - 1);
  size_t q = req->path.find('?');
  if (q != std::string::npos) req->path.resize(q);

  size_t content_length = 0;
  for (size_t pos = line_end + 2; pos < header_end;) {
    size_t eol = data.find("\r\n", pos);
    std::string h = data.substr(pos, eol - pos);
    if (strncasecmp(h.c_str(), "content-length:", 15) == 0)
      content_length = strtoul(h.c_str() + 15, nullptr, 10);
    pos = eol + 2;
  }
  if (content_length > kMaxRequestBytes) return false;
  req->body = data.substr(header_end + 4);
  while (req->body.size() < content_length) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n <= 0) return false;
    req->body.append(buf, n);
  }
  req->body.resize(content_length);
  return true;
}

bool Base64Decode(const std::string& in, std::string* out) {
  auto value = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
  };
  unsigned bits = 0;
  int nbits = 0;
  for (char c : in) {
    if (c == '=' || c == '\n' || c == '\r') continue;
    int v = value(c);
    if (v < 0) return false;
    bits = (bits << 6) | v;
    nbits += 6;
    if (nbits >= 8) {
      nbits -= 8;
      *out += char((bits >> nbits) & 0xFF);
    }
  }
  return true;
}

// Fills `msg` from "content": a string, or an array of
// {type: "text", text} and {type: "image_url", image_url: {url}} parts.
// Images must be data: URIs (data:image/png;base64,...); URLs aren't fetched.
bool ParseContent(const json::Value& content, nano::Message* msg, std::string* err) {
  if (content.is_string()) {
    msg->content = content.s;
    return true;
  }
  if (!content.is_array()) return true;
  for (const json::Value& part : content.items) {
    const std::string& type = part["type"].s;
    if (type == "text") {
      msg->content += part["text"].s;
    } else if (type == "image_url") {
      const json::Value& url_field = part["image_url"];
      const std::string& url = url_field.is_string() ? url_field.s : url_field["url"].s;
      size_t comma = url.find(',');
      if (url.compare(0, 5, "data:") != 0 || comma == std::string::npos ||
          url.rfind(";base64", comma) == std::string::npos) {
        *err = "image_url must be a base64 data: URI (data:image/png;base64,...).";
        return false;
      }
      std::string bytes;
      if (!Base64Decode(url.substr(comma + 1), &bytes)) {
        *err = "image_url has invalid base64 data.";
        return false;
      }
      msg->images.emplace_back();
      if (!nano::DecodeImage(bytes, &msg->images.back(), err)) return false;
    }
  }
  return true;
}

std::string TrimRight(std::string s) {
  while (!s.empty() && strchr(" \t\r\n", s.back())) s.pop_back();
  return s;
}

void HandleChat(int fd, const Request& req) {
  json::Value body;
  if (!json::Parse(req.body, &body) || body.type != json::Value::Type::kObject)
    return SendError(fd, 400, "Request body must be a JSON object.");
  const json::Value& msgs = body["messages"];
  if (!msgs.is_array() || msgs.items.empty())
    return SendError(fd, 400, "'messages' must be a non-empty array.");

  std::vector<nano::Message> messages;
  for (const json::Value& m : msgs.items) {
    const std::string& role = m["role"].s;
    nano::Role r;
    if (role == "system" || role == "developer") r = nano::Role::kSystem;
    else if (role == "user") r = nano::Role::kUser;
    else if (role == "assistant") r = nano::Role::kAssistant;
    else return SendError(fd, 400, "Unsupported message role: '" + role + "'.");
    nano::Message msg{r, "", {}};
    std::string err;
    if (!ParseContent(m["content"], &msg, &err)) return SendError(fd, 400, err);
    messages.push_back(std::move(msg));
  }

  nano::GenerateParams params;
  if (body["temperature"].is_number()) params.temperature = body["temperature"].n;
  if (body["top_k"].is_number()) params.top_k = body["top_k"].n;
  if (body["max_completion_tokens"].is_number())
    params.max_output_tokens = body["max_completion_tokens"].n;
  else if (body["max_tokens"].is_number())
    params.max_output_tokens = body["max_tokens"].n;
  bool stream = body["stream"].truthy();

  // A prompt that doesn't fit the context window crashes the model library, so
  // refuse it up front, before any response headers go out.
  {
    std::string terr;
    pthread_mutex_lock(&g_engine_mu);
    int prompt_tokens = g_engine.CountPromptTokens(messages, &terr);
    pthread_mutex_unlock(&g_engine_mu);
    if (prompt_tokens < 0) return SendError(fd, 500, terr);
    uint32_t ctx = g_engine.options().context_tokens;
    // The count is an estimate (a few tokens low), so keep some headroom.
    constexpr uint32_t kHeadroom = 32;
    if ((uint32_t)prompt_tokens + kHeadroom >= ctx) {
      Log("chat rejected: prompt is ~%d tokens, context is %u", prompt_tokens, ctx);
      return SendError(
          fd, 400,
          Fmt("This model's maximum context length is %u tokens, but the messages "
              "take about %d. Reduce the messages, or restart gnano-server with a "
              "larger --ctx.",
              ctx, prompt_tokens),
          "context_length_exceeded");
    }
  }

  pthread_mutex_lock(&g_id_mu);
  std::string id = Fmt("chatcmpl-%ld-%lu", (long)time(nullptr), g_next_id++);
  pthread_mutex_unlock(&g_id_mu);
  long created = time(nullptr);
  Log("chat %s: %zu messages, stream=%d", id.c_str(), messages.size(), stream);

  std::string chunk_prefix = "data: {\"id\":\"" + id +
                             "\",\"object\":\"chat.completion.chunk\"," +
                             Fmt("\"created\":%ld,", created) +
                             "\"model\":\"" + kModelName + "\",\"choices\":[";
  if (stream) {
    WriteAll(fd, std::string("HTTP/1.1 200 OK\r\n"
                             "Content-Type: text/event-stream\r\n"
                             "Cache-Control: no-cache\r\n") +
                     kCors + "Connection: close\r\n\r\n");
    WriteAll(fd, chunk_prefix +
                     "{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
                     "\"finish_reason\":null}]}\n\n");
  }

  std::string reply, err;
  nano::GenerateStats stats;
  pthread_mutex_lock(&g_engine_mu);
  bool ok = g_engine.Generate(
      messages, params,
      [&](const char* text) {
        if (!stream) {
          reply += text;
          return true;
        }
        // Stop generating if the client went away.
        return WriteAll(fd, chunk_prefix + "{\"index\":0,\"delta\":{\"content\":" +
                                json::Quote(text) + "},\"finish_reason\":null}]}\n\n");
      },
      &stats, &err);
  pthread_mutex_unlock(&g_engine_mu);

  // The reply also ends when prompt + reply fill the context window.
  bool hit_limit =
      (uint32_t)stats.output_tokens >= params.max_output_tokens ||
      (uint32_t)(stats.prompt_tokens + stats.output_tokens) + 1 >=
          g_engine.options().context_tokens;
  const char* finish = hit_limit ? "length" : "stop";
  Log("chat %s: %s, %d prompt + %d reply tokens, %.1f tok/s", id.c_str(),
      ok ? finish : err.c_str(), stats.prompt_tokens, stats.output_tokens,
      stats.decode_seconds > 0 ? stats.output_tokens / stats.decode_seconds : 0.0);

  std::string usage = Fmt(
      "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
      "\"total_tokens\":%d}",
      stats.prompt_tokens, stats.output_tokens,
      stats.prompt_tokens + stats.output_tokens);
  if (stream) {
    if (ok) {
      WriteAll(fd, chunk_prefix + "{\"index\":0,\"delta\":{},\"finish_reason\":\"" +
                       finish + "\"}]," + usage + "}\n\n");
    }
    WriteAll(fd, "data: [DONE]\n\n");
    return;
  }
  if (!ok) return SendError(fd, 500, err);
  SendJson(fd, 200,
           "{\"id\":\"" + id + "\",\"object\":\"chat.completion\"," +
               Fmt("\"created\":%ld,", created) + "\"model\":\"" + kModelName +
               "\",\"choices\":[{\"index\":0,\"message\":{\"role\":"
               "\"assistant\",\"content\":" +
               json::Quote(TrimRight(reply)) + "},\"finish_reason\":\"" + finish +
               "\"}]," + usage + "}");
}

void* HandleConnection(void* arg) {
  int fd = (int)(intptr_t)arg;
  Request req;
  if (!ReadRequest(fd, &req)) {
    SendError(fd, 400, "Malformed or oversized HTTP request.");
  } else if (req.method == "OPTIONS") {
    WriteAll(fd, std::string("HTTP/1.1 204 No Content\r\n") + kCors +
                     "Content-Length: 0\r\nConnection: close\r\n\r\n");
  } else if (req.method == "POST" && req.path == "/v1/chat/completions") {
    HandleChat(fd, req);
  } else if (req.method == "GET" && req.path == "/v1/models") {
    SendJson(fd, 200,
             std::string("{\"object\":\"list\",\"data\":[{\"id\":\"") + kModelName +
                 "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"google\"}]}");
  } else if (req.method == "GET" && req.path == "/healthz") {
    const nano::LoadOptions& o = g_engine.options();
    char ctx[16];
    snprintf(ctx, sizeof(ctx), "%u", o.context_tokens);
    SendJson(fd, 200,
             "{\"ok\":true,\"backend\":" + json::Quote(o.backend) +
                 ",\"model_dir\":" + json::Quote(o.model_dir) +
                 ",\"ctx\":" + ctx + "}");
  } else {
    SendError(fd, 404, "No route for " + req.method + " " + req.path);
  }
  close(fd);
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  const char* host = getenv("NANO_HOST") ? getenv("NANO_HOST") : "127.0.0.1";
  int port = getenv("NANO_PORT") ? atoi(getenv("NANO_PORT")) : 8765;
  uint32_t ctx = 0;  // 0: NANO_CTX, else 4096
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--port") && i + 1 < argc) {
      port = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--host") && i + 1 < argc) {
      host = argv[++i];
    } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
      ctx = static_cast<uint32_t>(atoi(argv[++i]));
    } else {
      fprintf(stderr,
              "usage: %s [--host 127.0.0.1] [--port 8765] [--ctx 4096]\n"
              "  --ctx N  context window in tokens (prompt + reply); a prompt\n"
              "           longer than this crashes the server\n"
              "Same NANO_* environment variables as gnano, plus NANO_HOST, "
              "NANO_PORT and NANO_CTX.\n",
              argv[0]);
      return 2;
    }
  }
  signal(SIGPIPE, SIG_IGN);  // a client hanging up must not kill the server

  nano::LoadOptions opts;
  opts.context_tokens = ctx;
  std::string err;
  if (!nano::ResolveLoadOptions(&opts, &err) || !g_engine.Load(opts, &err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }

  int srv = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
    fprintf(stderr, "bad --host %s (use an IPv4 address)\n", host);
    return 1;
  }
  if (bind(srv, (sockaddr*)&addr, sizeof(addr)) != 0 || listen(srv, 64) != 0) {
    fprintf(stderr, "cannot listen on %s:%d: %s\n", host, port, strerror(errno));
    return 1;
  }
  Log("READY: OpenAI-compatible API on http://%s:%d/v1 (model \"%s\")", host,
      port, kModelName);

  for (;;) {
    int fd = accept(srv, nullptr, nullptr);
    if (fd < 0) continue;
    pthread_t t;
    if (pthread_create(&t, nullptr, HandleConnection, (void*)(intptr_t)fd) != 0) {
      close(fd);
      continue;
    }
    pthread_detach(t);
  }
}
