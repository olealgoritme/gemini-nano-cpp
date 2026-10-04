// Minimal JSON: just enough to read OpenAI requests and write replies.
// Header-only; no exceptions (parse errors return false).

#ifndef JSON_H_
#define JSON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

namespace json {

struct Value {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  Type type = Type::kNull;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Value> items;                              // kArray
  std::vector<std::pair<std::string, Value>> members;    // kObject

  // Member lookup; returns a null Value if missing or not an object.
  const Value& operator[](const char* key) const {
    static const Value kNull;
    if (type == Type::kObject) {
      for (const auto& [k, v] : members)
        if (k == key) return v;
    }
    return kNull;
  }
  bool is_null() const { return type == Type::kNull; }
  bool is_string() const { return type == Type::kString; }
  bool is_number() const { return type == Type::kNumber; }
  bool is_array() const { return type == Type::kArray; }
  bool truthy() const { return type == Type::kBool && b; }
};

class Parser {
 public:
  explicit Parser(const std::string& text) : p_(text.c_str()), end_(p_ + text.size()) {}

  bool Parse(Value* out) {
    if (!ParseValue(out, 0)) return false;
    SkipSpace();
    return p_ == end_;
  }

 private:
  void SkipSpace() {
    while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
  }

  bool Literal(const char* word) {
    size_t len = strlen(word);
    if (size_t(end_ - p_) < len || memcmp(p_, word, len) != 0) return false;
    p_ += len;
    return true;
  }

  static void AppendUtf8(std::string* s, unsigned cp) {
    if (cp < 0x80) {
      *s += char(cp);
    } else if (cp < 0x800) {
      *s += char(0xC0 | (cp >> 6));
      *s += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      *s += char(0xE0 | (cp >> 12));
      *s += char(0x80 | ((cp >> 6) & 0x3F));
      *s += char(0x80 | (cp & 0x3F));
    } else {
      *s += char(0xF0 | (cp >> 18));
      *s += char(0x80 | ((cp >> 12) & 0x3F));
      *s += char(0x80 | ((cp >> 6) & 0x3F));
      *s += char(0x80 | (cp & 0x3F));
    }
  }

  bool Hex4(unsigned* out) {
    if (end_ - p_ < 4) return false;
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = *p_++;
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else return false;
    }
    *out = v;
    return true;
  }

  bool ParseString(std::string* out) {
    if (p_ >= end_ || *p_ != '"') return false;
    ++p_;
    while (p_ < end_ && *p_ != '"') {
      char c = *p_++;
      if (c != '\\') {
        *out += c;
        continue;
      }
      if (p_ >= end_) return false;
      char e = *p_++;
      switch (e) {
        case '"': *out += '"'; break;
        case '\\': *out += '\\'; break;
        case '/': *out += '/'; break;
        case 'b': *out += '\b'; break;
        case 'f': *out += '\f'; break;
        case 'n': *out += '\n'; break;
        case 'r': *out += '\r'; break;
        case 't': *out += '\t'; break;
        case 'u': {
          unsigned cp;
          if (!Hex4(&cp)) return false;
          // Surrogate pair.
          if (cp >= 0xD800 && cp < 0xDC00 && end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
            p_ += 2;
            unsigned lo;
            if (!Hex4(&lo)) return false;
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          AppendUtf8(out, cp);
          break;
        }
        default: return false;
      }
    }
    if (p_ >= end_) return false;
    ++p_;
    return true;
  }

  bool ParseValue(Value* v, int depth) {
    if (depth > 64) return false;
    SkipSpace();
    if (p_ >= end_) return false;
    switch (*p_) {
      case 'n': v->type = Value::Type::kNull; return Literal("null");
      case 't': v->type = Value::Type::kBool; v->b = true; return Literal("true");
      case 'f': v->type = Value::Type::kBool; v->b = false; return Literal("false");
      case '"': v->type = Value::Type::kString; return ParseString(&v->s);
      case '[': {
        ++p_;
        v->type = Value::Type::kArray;
        SkipSpace();
        if (p_ < end_ && *p_ == ']') { ++p_; return true; }
        for (;;) {
          v->items.emplace_back();
          if (!ParseValue(&v->items.back(), depth + 1)) return false;
          SkipSpace();
          if (p_ < end_ && *p_ == ',') { ++p_; continue; }
          if (p_ < end_ && *p_ == ']') { ++p_; return true; }
          return false;
        }
      }
      case '{': {
        ++p_;
        v->type = Value::Type::kObject;
        SkipSpace();
        if (p_ < end_ && *p_ == '}') { ++p_; return true; }
        for (;;) {
          SkipSpace();
          std::string key;
          if (!ParseString(&key)) return false;
          SkipSpace();
          if (p_ >= end_ || *p_ != ':') return false;
          ++p_;
          v->members.emplace_back(std::move(key), Value());
          if (!ParseValue(&v->members.back().second, depth + 1)) return false;
          SkipSpace();
          if (p_ < end_ && *p_ == ',') { ++p_; continue; }
          if (p_ < end_ && *p_ == '}') { ++p_; return true; }
          return false;
        }
      }
      default: {
        char* num_end;
        std::string tmp(p_, end_ - p_ < 64 ? end_ - p_ : 64);
        v->n = strtod(tmp.c_str(), &num_end);
        if (num_end == tmp.c_str()) return false;
        v->type = Value::Type::kNumber;
        p_ += num_end - tmp.c_str();
        return true;
      }
    }
  }

  const char* p_;
  const char* end_;
};

inline bool Parse(const std::string& text, Value* out) {
  return Parser(text).Parse(out);
}

// Returns `s` as a quoted JSON string.
inline std::string Quote(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += char(c);
        }
    }
  }
  return out + "\"";
}

}  // namespace json

#endif  // JSON_H_
