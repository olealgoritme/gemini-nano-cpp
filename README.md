# gemini-nano-cpp

Run Google's **Gemini Nano**, the small AI model built into Google Chrome,
**directly from the command line, without Chrome running**. It runs on your
graphics card (GPU) or your processor (CPU). It can also act as a local
**OpenAI-compatible API**, so existing AI apps and libraries can use it.

```
$ gnano write a haiku about Linux
Open source and free,
Powering servers, desktops,
Flexibility reigns.
```

Measured on an RTX 5090 and a 32-core CPU:

| | GPU model | CPU model |
|---|---|---|
| Speed | ~64 tokens/s | ~20 tokens/s |
| Model load | ~1.5 s | ~0.1 s |
| RAM used | ~0.5 GB (the model sits in VRAM) | ~1.9 GB |

---

## How it works (the short version)

Chrome downloads an AI model called Gemini Nano onto your computer. You can't
load that file with normal AI tools (llama.cpp, Ollama, ...) because it isn't
in any standard format. The code that knows how to run it lives inside Chrome.

That code isn't buried in the main `chrome` program, though. It's a separate
file that ships with Chrome:

```
/opt/google/chrome/liboptimization_guide_internal.so
```

Chrome loads this file and calls a function in it named `GetChromeMLAPIV2`,
which hands back a list of functions such as "load model", "start chat" and
"generate text". This project does the same thing from a small C++ program,
with no browser involved.

For the GPU, the model also needs **Dawn**, Google's open-source WebGPU
library. Chrome keeps its copy of Dawn inside the `chrome` program, so this
project builds the exact same version of Dawn from source.

```
          one-time setup                            every time after
 ┌────────────────────────────────────┐   ┌──────────────────────────────────┐
 │ gnano-download                     │   │ gnano / gnano-server             │
 │  → starts hidden Chrome            │   │  → loads Google's library        │
 │  → clicks "download" for you       │   │  → loads our Dawn build (GPU)    │
 │  → saves the model and a copy of   │   │  → loads the model               │
 │    Google's library                │   │  → answers                       │
 └────────────────────────────────────┘   └──────────────────────────────────┘
           needs Chrome                           Chrome NOT needed
```

---

## What you need

| | |
|---|---|
| **Computer** | Linux, 64-bit Intel/AMD (x86-64). Not ARM, not Alpine. |
| **For the GPU model** | A graphics card with a working **Vulkan** driver and more than 4 GB of VRAM |
| **For the CPU model** | 16 GB RAM and at least 4 CPU cores |
| **Disk** | ~4 GB per model, ~3 GB for build tools and sources. Chrome wants ~22 GB free before it agrees to download. |
| **Google Chrome** | Version **154.0.8037.57**, only for the one-time download (see [Chrome versions](#chrome-versions)). Chromium won't work. |
| **Tools** | `python3`, `git`, `curl`, `tar`, `xz`, `cmake`, `ninja` |

Install the tools on Ubuntu/Debian:

```sh
sudo apt install python3 git curl xz-utils cmake ninja-build
```

On Fedora: `sudo dnf install python3 git curl xz cmake ninja-build`.
On Arch: `sudo pacman -S python git curl xz cmake ninja`.

Check your Chrome version:

```sh
google-chrome --version
```

If you don't have Chrome, get it from <https://www.google.com/chrome/>.

---

## Step by step

### 1. Get the code

```sh
git clone https://github.com/olealgoritme/gemini-nano-cpp.git
cd gemini-nano-cpp
```

### 2. Build (one time)

```sh
make
```

The first run downloads and builds its tools in `third_party/`: Chromium's
compiler, its C++ library headers, and Dawn. That takes a few minutes. Later
builds take seconds. The programs end up in `build/release/`.

### 3. Install

```sh
sudo make install
```

This installs three commands into `/usr/local/bin`:

| Command | What it does |
|---|---|
| `gnano` | ask a question from the terminal |
| `gnano-server` | start the OpenAI-compatible API |
| `gnano-download` | download a model |

No sudo? Use `make install PREFIX=~/.local` instead, which installs into
`~/.local/bin`. To remove it again: `sudo make uninstall`.

### 4. Download a model (one time, ~4 GB)

There are two versions of the model, and you can download either or both:

```sh
gnano-download          # GPU model
gnano-download --cpu    # CPU model
```

(Without installing: `make download` or `make download-cpu`.)

You'll see a progress bar:

```
Using Google Chrome 154.0.8037.57 (/usr/bin/google-chrome-stable)
Starting Chrome (hidden), profile: /home/you/.local/share/gemini-nano/profile-gpu
Model availability: downloadable
Clicked; download starting (about 4 GB, this can take a while)...
  [##############                          ]  35.2%  (downloading)
```

When it finishes, it prints `Model ready: ...`.

- **Which one?** Use the GPU model if you have a decent graphics card, since
  it's about 3× faster. Otherwise use the CPU model.
- **Where does it go?** Into `~/.local/share/gemini-nano/`, once per user. It
  doesn't matter which folder you run it from.
- **It's safe to stop** with Ctrl-C. Run it again to continue.
- **Stuck or failing?** Add `--show`. This opens a visible Chrome window doing
  the same thing, which sometimes helps.
- Your normal Chrome is never touched. The download uses its own Chrome
  profile inside that folder.

### 5. Ask it something

```sh
gnano hey whats up                  # ask anything (quotes are optional)
echo "summarize: ..." | gnano       # or pipe text in
gnano -v hi                         # also show logs and speed
gnano --image cat.jpg what is this  # show it a picture (JPEG, PNG, GIF, BMP)
gnano --image a.png --image b.png compare these
```

That's it. Chrome doesn't need to be open, or even installed anymore, because
the download also saved a copy of Google's library next to the models.

---

## What's inside the model?

Gemini Nano v3 turns out to be a **Gemma 3n E2B**-architecture model, the
same design as Google's open-weights Gemma 3n: 30 layers, a 2048-wide hidden
state, "per-layer embeddings", a MobileNet-V5 image encoder and the same
262,144-token tokenizer. It's not the same *weights* as Gemma 3n, and Google
adds about 6,200 internal tokens (object detection, video, "visual reasoning").

This was established without decrypting anything. The details, and how each
value is known, are in [`src/gemini_nano_model.h`](src/gemini_nano_model.h).
You can check it yourself:

```sh
make inspect      # the model files' layout
make verify       # ~20 checks against the public Gemma 3n E2B config and tokenizer
gnano --info                        # what the library reports, as JSON
gnano --dump-tokenizer vocab.json   # the full vocabulary
gnano --tokenize hello world        # text -> token ids
```

`make verify` downloads Gemma 3n's public `config.json` and `tokenizer.json`
once into `~/.cache/gemini-nano`. For the tokenizer comparison it needs
`pip install tokenizers`; without that, that one check is skipped.

The part of each model file that lists the individual layers and tensors is
encrypted by Google, so this project doesn't go further than that.

---

## OpenAI-compatible API

```sh
./gnano-server            # or: gnano-server
```

```
20:11:02 READY: OpenAI-compatible API on http://127.0.0.1:8765/v1 (model "gemini-nano")
```

The model stays loaded, so every request is fast. Try it with curl:

```sh
curl http://localhost:8765/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "gemini-nano",
  "messages": [{"role": "user", "content": "Write a haiku about local AI."}]
}'
```

Or with any OpenAI library, for example Python:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8765/v1", api_key="not-needed")
reply = client.chat.completions.create(
    model="gemini-nano",
    messages=[{"role": "user", "content": "hello"}],
)
print(reply.choices[0].message.content)
```

What's supported:

| | |
|---|---|
| `POST /v1/chat/completions` | `system`, `user` and `assistant` messages (full conversation history), images, `stream: true` (live, word by word), `temperature`, `top_k`, `max_tokens` / `max_completion_tokens` |
| `GET /v1/models` | lists `gemini-nano` |
| `GET /healthz` | which backend and model are loaded |

Options: `--port 8765` and `--host 127.0.0.1`, or `NANO_PORT` / `NANO_HOST`.
Use `--host 0.0.0.0` to allow other computers on your network to connect
(there's no password, so only do this on a network you trust).

Images go in the OpenAI way, as an `image_url` part with a base64 `data:` URI
(web links aren't downloaded):

```python
import base64
url = "data:image/jpeg;base64," + base64.b64encode(open("cat.jpg", "rb").read()).decode()
reply = client.chat.completions.create(model="gemini-nano", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": url}},
    {"type": "text", "text": "What is in this picture?"},
]}])
```

Requests are answered one at a time; others wait their turn. Tool calling and
audio aren't supported yet.

---

## Settings

Everything is found automatically. To change something, set these
environment variables:

| Variable | What it is | Default |
|---|---|---|
| `NANO_BACKEND` | `gpu` or `cpu` | `gpu` if a GPU model is downloaded, else `cpu` |
| `NANO_HOME` | folder holding the models and Google's library | `~/.local/share/gemini-nano` |
| `NANO_MODEL_DIR` | folder containing `weights.bin` | newest model in `profile-<backend>/` |
| `NANO_LIB` | path to `liboptimization_guide_internal.so` | the copy in `NANO_HOME/lib/`, else `/opt/google/chrome/` |
| `NANO_DAWN_LIB` | path to our Dawn build | `libnano_dawn.so` next to the program |

For example, to use the CPU model even though you have both:

```sh
NANO_BACKEND=cpu gnano hello
```

`download-model.py` reads `NANO_CHROME` if Chrome isn't in a standard place.

**Note:** each backend needs its own model. Running `NANO_BACKEND=cpu` with
the GPU model fails with `TF_LITE_PREFILL_DECODE model is expected to exist`.

---

## Chrome versions

The library's API is **not stable**. Google changes it between Chrome
releases. Everything here is pinned to Chromium **154.0.8037.57**:

- the header files in `chromium/include/`
- the compiler, C++ library and Dawn revisions in the `Makefile`
- the Dawn version check in `src/engine.cc`

`gnano-download` **saves a copy of the library next to the models**, so a
later Chrome auto-update won't break a working setup. To support a new Chrome
version, update all of the above to the matching Chromium tag.

---

## Why the special compiler?

Google's library passes C++ objects (text strings, callback functions) back and
forth. For that to work, our program has to lay out those objects in memory
exactly like Chrome does. Chrome uses its own build of the C++ standard library
(libc++, "ABI v2", namespace `std::__Cr`), so `make` downloads:

- **Chromium's libc++ headers**, at the exact revision Chrome 154 uses
- **Chromium's clang compiler** (those headers need clang 21 or newer)

It also downloads `stb_image`, a small public-domain image decoder.

Both go into `third_party/` and are only used for building.

---

## For developers

```sh
make DEBUG=1              # unoptimized build for gdb, in build/debug/
make clean                # remove build/
make distclean            # also remove the downloaded tools in third_party/
```

You can run the programs without installing: `build/release/gnano hi`.

| Path | What it is |
|---|---|
| `src/engine.cc`, `src/engine.h` | Everything that talks to Google's library: loading, chatting |
| `src/main.cc` | `gnano`, the command-line program |
| `src/image.cc`, `src/image.h` | Decodes image files (with stb_image) for `--image` |
| `src/gemini_nano_model.h` | What the model is: architecture, tokenizer, file layout, and how each value is known |
| `src/server.cc`, `src/json.h` | `gnano-server`, the OpenAI-compatible API |
| `src/dawn/` | Builds Dawn and exposes its function table as `libnano_dawn.so` |
| `scripts/download-model.py` | `gnano-download`, with the automated click |
| `scripts/inspect-model.py` | Maps the model files' layout (`make inspect`) |
| `scripts/verify-model.py` | Tests the model against the public Gemma 3n E2B (`make verify`) |
| `chromium/include/` | Chromium's headers describing the library's API (plus two tiny stand-ins for Dawn headers) |
| `chromium/libcxx-config/` | Chromium's settings for its C++ standard library |
| `chromium/ref/` | Chromium source files showing how Chrome itself calls the library |
| `build/` | Build output (not in git) |
| `third_party/` | Downloaded build tools: clang, libc++, Dawn (not in git) |

Where things live after installing:

| | |
|---|---|
| `/usr/local/bin/gnano*` | the commands (links) |
| `/usr/local/lib/gemini-nano/` | the programs and our Dawn build |
| `~/.local/share/gemini-nano/` | your models (`profile-gpu/`, `profile-cpu/`) and the copy of Google's library (`lib/`) |

---

## Legal

The code in this repo is MIT-licensed, © 2026 Ole Algoritme. See
[LICENSE](LICENSE). **Google's library
(`liboptimization_guide_internal.so`) and the model files are proprietary.**
This repo doesn't contain them. They come from your own Chrome install. Don't
redistribute them.

Files in `chromium/` come from the Chromium project (BSD license) and LLVM
(Apache 2.0 with LLVM exception). Dawn is BSD-licensed and is downloaded at
build time.
