#!/usr/bin/env python3
"""Test the downloaded Gemini Nano models against the public Gemma 3n E2B
architecture, which they appear to be based on.

Evidence comes only from openly readable sources: the model files' container
layout (inspect-model.py), what Chrome's model library reports through its
API (gnano --info / --dump-tokenizer / --tokenize), and Gemma 3n E2B's public
config.json and tokenizer.json (downloaded once into ~/.cache/gemini-nano).

Each check prints PASS, FAIL, SKIP (not testable here) or INFO (a measurement
with no pass/fail). The exit code is 1 if anything failed.

Usage:
    verify-model.py                 # uses gnano from build/release, next to
                                    # this script, or PATH
    verify-model.py --gnano PATH
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
# Open re-upload of google/gemma-3n-E2B-it (Google's own copy needs a login);
# its config.json and tokenizer.json are the same files.
REFERENCE_REPO = "https://huggingface.co/unsloth/gemma-3n-E2B-it/resolve/main"
CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                     "gemini-nano", "reference")

SAMPLES = [
    "Hello, world!",
    "Hei på deg! Hvordan går det med deg i dag? Æ, ø og å.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "東京は日本の首都です。 Привет, мир! مرحبا بالعالم",
    "Emoji test 👋🏽🚀 and numbers 3.14159, 1,000,000 and 2026-10-04.",
    "   leading spaces, trailing tabs\t\t and\n\nnewlines  ",
]

results = []


def report(status, name, detail=""):
    results.append(status)
    print(f"  {status:4}  {name}" + (f"\n        {detail}" if detail else ""))


def check(ok, name, detail=""):
    report("PASS" if ok else "FAIL", name, detail)


def find_gnano():
    if "--gnano" in sys.argv:
        return sys.argv[sys.argv.index("--gnano") + 1]
    for path in (os.path.join(HERE, "..", "build", "release", "gnano"),
                 os.path.join(HERE, "gnano")):
        if os.access(path, os.X_OK):
            return os.path.abspath(path)
    return shutil.which("gnano") or sys.exit("gnano not found: run make, or pass --gnano PATH")


def gnano(binary, backend, *args, stdin=None):
    env = dict(os.environ, NANO_BACKEND=backend)
    out = subprocess.run([binary, *args], env=env, input=stdin, capture_output=True,
                         text=True, check=True)
    return out.stdout


def reference(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print(f"  (downloading public Gemma 3n E2B {name} into {CACHE})")
        urllib.request.urlretrieve(f"{REFERENCE_REPO}/{name}", path + ".part")
        os.rename(path + ".part", path)
    return path


def public_vocab(tokenizer_json):
    """Public tokenizer.json -> {id: token bytes}, in our notation."""
    tj = json.load(open(tokenizer_json))
    by_id = {i: t for t, i in tj["model"]["vocab"].items()}
    for added in tj.get("added_tokens", []):
        by_id[added["id"]] = added["content"]
    out = {}
    for i, t in by_id.items():
        m = re.fullmatch(r"<0x([0-9A-F]{2})>", t)  # byte-fallback tokens
        out[i] = bytes([int(m.group(1), 16)]) if m else t.replace("▁", " ").encode()
    return out


def our_vocab(dump):
    return [bytes.fromhex(t["hex"]) if isinstance(t, dict) else t.encode()
            for t in dump["vocab"]]


# Gemma's conversation-turn markers, which Nano names differently.
TURN_MARKERS = {105: (b"<start_of_turn>", b"<ctrl99>"), 106: (b"<end_of_turn>", b"<ctrl100>")}


def verify_tokenizer(binary, backend, cfg, gen_cfg, tokenizer_json):
    print(f"\nTokenizer ({backend} model, via the library's API)")
    info = json.loads(gnano(binary, backend, "--info"))
    tok = info["tokenizer"]
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        gnano(binary, backend, "--dump-tokenizer", f.name)
        ours = our_vocab(json.load(open(f.name)))
    text = cfg["text_config"]

    check(tok["vocab_size"] == text["vocab_size_per_layer_input"],
          f"vocabulary size = {text['vocab_size_per_layer_input']:,} (Gemma 3n text vocabulary)",
          f"library reports {tok['vocab_size']:,}")
    # generation_config.json decides when generation stops.
    pub_eos = gen_cfg["eos_token_id"]
    pub_eos = sorted(pub_eos if isinstance(pub_eos, list) else [pub_eos])
    check(sorted(tok["eos_ids"]) == pub_eos, f"end-of-sequence ids = {pub_eos}",
          f"library reports {sorted(tok['eos_ids'])}")
    for key, want in (("boi_token_id", "<start_of_image>"), ("boa_token_id", "<start_of_audio>")):
        i = cfg[key]
        check(ours[i] == want.encode(), f"{key} {i} is {want}", f"Nano has {ours[i]!r}")

    pub = public_vocab(tokenizer_json)
    named = [i for i in range(len(ours)) if not re.fullmatch(rb"<unused\d+>", pub.get(i, b"<unused0>"))]
    differ = [i for i in named if ours[i] != pub[i]]
    unexpected = [i for i in differ if TURN_MARKERS.get(i) != (pub[i], ours[i])]
    check(not unexpected,
          f"every public Gemma 3n token is identical in Nano ({len(named) - len(differ):,} of "
          f"{len(named):,}), except the 2 turn markers",
          "105 <start_of_turn> / 106 <end_of_turn> are named <ctrl99> / <ctrl100> in Nano"
          + (f"; unexpected differences at ids {unexpected[:10]}" if unexpected else ""))
    renamed = len(ours) - len(named)
    report("INFO", f"{renamed:,} slots Gemma 3n leaves as <unusedN> are named in Nano",
           "e.g. " + ", ".join(ours[i].decode(errors="replace") for i in
                               (7, 256001, 257500, 262140, 262142)))

    if importlib.util.find_spec("tokenizers") is None:
        report("SKIP", "text is split into the same tokens as the public tokenizer",
               "pip install tokenizers  to run this check")
        return
    from tokenizers import Tokenizer
    public = Tokenizer.from_file(tokenizer_json)
    bad = []
    for s in SAMPLES:
        lib = json.loads(gnano(binary, backend, "--tokenize", stdin=s))
        if lib != public.encode(s, add_special_tokens=False).ids:
            bad.append(s)
    check(not bad, f"{len(SAMPLES)} sample texts split into the same tokens as the public tokenizer",
          f"differs for: {bad!r}" if bad else "English, Norwegian, code, CJK/Cyrillic/Arabic, emoji, whitespace")


def test_png(path):
    """Writes a small solid-color PNG (no imaging libraries needed)."""
    import struct
    import zlib
    w = h = 64
    rows = b"".join(b"\x00" + b"\xdc\x14\x14" * w for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def prompt_tokens(binary, backend, *args):
    env = dict(os.environ, NANO_BACKEND=backend)
    out = subprocess.run([binary, "-v", *args], env=env, capture_output=True, text=True)
    m = re.search(r"prompt (\d+) tokens", out.stderr)
    return int(m.group(1)) if m else None


def verify_image_input(binary, backend, cfg):
    print(f"\nImage input ({backend} model, by running it)")
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "red.png")
        test_png(png)
        q = "What color is this?"
        base = prompt_tokens(binary, backend, q)
        one = prompt_tokens(binary, backend, "--image", png, q)
        two = prompt_tokens(binary, backend, "--image", png, "--image", png, q)
    if None in (base, one, two):
        check(False, "image input works", "gnano --image failed")
        return
    per_image = cfg["vision_soft_tokens_per_image"] + 2  # + <start_of_image>, <end_of_image>
    check(one - base == per_image and two - one == per_image,
          f"each image is {cfg['vision_soft_tokens_per_image']} image tokens + 2 markers = {per_image}",
          f"prompt tokens: {base} text only, {one} with 1 image, {two} with 2")


def verify_cpu_layout(info, cfg):
    print("\nArchitecture vs. section sizes (CPU model file)")
    t = cfg["text_config"]
    sec = {s["name"]: s["size"] for s in info["sections"]}
    vocab, hidden = t["vocab_size_per_layer_input"], t["hidden_size"]
    layers, ple = t["num_hidden_layers"], t["hidden_size_per_layer_input"]

    # Embedding tables are vocab rows of int4 values plus a small per-row
    # overhead (scales). Solve for the row width and compare.
    def table(name, width, label):
        size = sec[name]
        payload = vocab * width // 2  # 4 bits per value
        overhead = (size - payload) / vocab
        check(0 <= overhead < 32,
              f"{name}: {vocab:,} x {label} = {width:,} values at 4 bits",
              f"expected {payload:,} B + small per-row overhead; file has {size:,} B "
              f"-> {overhead:.1f} B/row overhead")
        return overhead

    o1 = table("TF_LITE_EMBEDDER", hidden, f"hidden_size {hidden}")
    o2 = table("TF_LITE_PER_LAYER_EMBEDDER", layers * ple,
               f"{layers} layers x {ple} per-layer dims")
    check(abs(o1 - o2) < 0.1, "both embedding tables use the same row format",
          f"{o1:.2f} vs {o2:.2f} bytes of overhead per row")
    for alt in (35,):  # Gemma 3n E4B
        size = vocab * alt * ple // 2
        check(sec["TF_LITE_PER_LAYER_EMBEDDER"] < size,
              f"rules out the {alt}-layer E4B variant",
              f"E4B would need at least {size:,} B")

    vis = sec.get("TF_LITE_VISION_ENCODER", 0)
    params = vis * 2 / 1e6
    check(250 <= params <= 350, "TF_LITE_VISION_ENCODER fits MobileNet-V5-300M at 4 bits",
          f"{vis:,} B = {params:.0f}M values at 4 bits (public: {cfg['vision_config'].get('architecture')})")
    report("INFO" if "TF_LITE_AUDIO_ENCODER" not in sec else "PASS",
           "no audio encoder section in the CPU model file",
           "Gemma 3n has an audio tower; this CPU file has no audio section")

    # Decoder: count the weights the public architecture implies.
    heads, kv, hd, ff = (t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"],
                         t["intermediate_size"])
    ff = ff[0] if isinstance(ff, list) else ff
    per_layer = (hidden * heads * hd * 2 + hidden * kv * hd * 2  # q, o / k, v
                 + 3 * hidden * ff                                # gated MLP
                 + 2 * hidden * t["laurel_rank"]                  # LAuReL
                 + 2 * hidden * ple)                              # per-layer input
    total = layers * per_layer + hidden * layers * ple + 6 * hidden * hidden  # + AltUp
    bits = sec["TF_LITE_PREFILL_DECODE"] * 8 / total
    report("INFO", "TF_LITE_PREFILL_DECODE size vs. the public decoder",
           f"public config implies ~{total / 1e9:.2f}B weights; {sec['TF_LITE_PREFILL_DECODE']:,} B "
           f"= {bits:.1f} bits/weight (mixed 4/8-bit, or extra tensors; layout unknown)")


def main():
    binary = find_gnano()
    print(f"Using {binary}")
    spec = importlib.util.spec_from_file_location("inspect_model", os.path.join(HERE, "inspect-model.py"))
    inspect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspect)

    cfg = json.load(open(reference("config.json")))
    gen_cfg = json.load(open(reference("generation_config.json")))
    tokenizer_json = reference("tokenizer.json")

    models = inspect.find_models()
    if not models:
        sys.exit("No models downloaded yet (gnano-download).")
    for model_dir in models:
        backend = "cpu" if "/profile-cpu/" in model_dir else "gpu"
        info = inspect.inspect(model_dir)
        print(f"\n===== {backend.upper()} model {info['manifest']['version']} =====")
        verify_tokenizer(binary, backend, cfg, gen_cfg, tokenizer_json)
        verify_image_input(binary, backend, cfg)
        caps = json.loads(gnano(binary, backend, "--info"))["capabilities"]
        print(f"\nCapabilities ({backend} model, via the library's API)")
        if caps is None:
            report("SKIP", "image/audio input", "the library can't report this for this file")
        else:
            check(caps["image_input"] and caps["audio_input"],
                  "image and audio input supported (Gemma 3n has vision and audio towers)",
                  f"library reports {caps}")
        if info["sections"]:
            verify_cpu_layout(info, cfg)
        else:
            print("\nArchitecture vs. section sizes")
            report("SKIP", "section sizes", "this model file has no readable container")

    failed = results.count("FAIL")
    print(f"\n{results.count('PASS')} passed, {failed} failed, "
          f"{results.count('SKIP')} skipped, {results.count('INFO')} info")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
