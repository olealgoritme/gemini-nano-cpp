#include "image.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

// stb_image (public domain), downloaded into third_party/stb by the Makefile.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

namespace nano {

bool DecodeImage(const std::string& bytes, Image* out, std::string* err) {
  int w, h, channels;
  stbi_uc* rgba = stbi_load_from_memory(
      reinterpret_cast<const stbi_uc*>(bytes.data()), (int)bytes.size(), &w, &h,
      &channels, 4);
  if (!rgba) {
    *err = std::string("cannot decode image: ") + stbi_failure_reason();
    return false;
  }
  out->width = w;
  out->height = h;
  out->bgra.resize((size_t)w * h * 4);
  for (size_t i = 0; i < out->bgra.size(); i += 4) {
    unsigned a = rgba[i + 3];
    out->bgra[i + 0] = (uint8_t)(rgba[i + 2] * a / 255);  // B
    out->bgra[i + 1] = (uint8_t)(rgba[i + 1] * a / 255);  // G
    out->bgra[i + 2] = (uint8_t)(rgba[i + 0] * a / 255);  // R
    out->bgra[i + 3] = (uint8_t)a;
  }
  stbi_image_free(rgba);
  return true;
}

bool ReadFile(const std::string& path, std::string* out, std::string* err) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    *err = path + ": " + strerror(errno);
    return false;
  }
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
  bool ok = !ferror(f);
  fclose(f);
  if (!ok) *err = path + ": read error";
  return ok;
}

}  // namespace nano
