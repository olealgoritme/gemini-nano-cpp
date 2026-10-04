// Decoding images (JPEG, PNG, GIF, BMP, ...) for the model's image input.

#ifndef IMAGE_H_
#define IMAGE_H_

#include <stdint.h>

#include <string>
#include <vector>

namespace nano {

// Pixels in Skia's native 32-bit format on Linux (kN32 = BGRA), with
// premultiplied alpha, rows packed without padding.
struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> bgra;
};

// Decodes an encoded image file's bytes.
bool DecodeImage(const std::string& bytes, Image* out, std::string* err);

// Reads a whole file.
bool ReadFile(const std::string& path, std::string* out, std::string* err);

}  // namespace nano

#endif  // IMAGE_H_
