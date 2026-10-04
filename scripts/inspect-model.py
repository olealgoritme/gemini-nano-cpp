#!/usr/bin/env python3
"""Map the container layout of downloaded Gemini Nano model files.

Reads only what is stored openly: manifest.json, the file list, and the ZIP
entry headers of the model file (the CPU model is a ZIP of sections such as
TF_LITE_PREFILL_DECODE). The start of each section is encrypted and is left
alone; this script never tries to decrypt anything.

Usage:
    inspect-model.py                 # every model in ~/.local/share/gemini-nano
    inspect-model.py <model-dir>...  # folders holding weights.bin
    inspect-model.py --json          # machine-readable (used by verify-model.py)
"""

import glob
import json
import mmap
import os
import struct
import sys

DATA = os.environ.get("NANO_HOME") or os.path.join(
    os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"),
    "gemini-nano")


def find_models():
    return sorted(os.path.dirname(p) for p in glob.glob(
        os.path.join(DATA, "profile-*", "OptGuideOnDeviceModel", "*", "weights.bin")))


def zip_sections(path):
    """Walks the ZIP local file headers; returns [] if there are none."""
    sections = []
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        off = m.find(b"PK\x03\x04", 0, 4096)
        while 0 <= off < len(m) and m[off:off + 4] == b"PK\x03\x04":
            (_, flags, method, _, _, _, csize, usize, nlen,
             xlen) = struct.unpack("<HHHHHIIIHH", m[off + 4:off + 30])
            name = m[off + 30:off + 30 + nlen].decode("utf-8", "replace")
            extra = m[off + 30 + nlen:off + 30 + nlen + xlen]
            p = 0
            while p + 4 <= len(extra):  # ZIP64 sizes live in extra field 1
                hid, hlen = struct.unpack("<HH", extra[p:p + 4])
                if hid == 1:
                    vals, i = extra[p + 4:p + 4 + hlen], 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[i:i + 8])[0]
                        i += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[i:i + 8])[0]
                p += 4 + hlen
            data = off + 30 + nlen + xlen
            head = m[data:data + 8]
            sections.append({
                "name": name, "offset": data, "size": usize,
                "stored": method == 0,
                # A plain TFLite flatbuffer has "TFL3" at byte 4.
                "header_readable": head[4:8] == b"TFL3",
            })
            nxt = data + csize
            if m[nxt:nxt + 4] != b"PK\x03\x04":  # entries are padded/aligned
                nxt = m.find(b"PK\x03\x04", nxt, nxt + (1 << 20))
            off = nxt
        m.close()
    return sections


def inspect(model_dir):
    weights = os.path.join(model_dir, "weights.bin")
    info = {"model_dir": model_dir, "files": {}, "manifest": None}
    for name in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            info["files"][name] = os.path.getsize(path)
    try:
        info["manifest"] = json.load(open(os.path.join(model_dir, "manifest.json")))
    except (OSError, ValueError):
        pass
    info["sections"] = zip_sections(weights)
    return info


def print_report(info):
    man = info["manifest"] or {}
    spec = man.get("BaseModelSpec", {})
    print(f"== {info['model_dir']}")
    print(f"   version {man.get('version')}, base model {spec.get('name')} "
          f"{spec.get('version')}, performance hints {spec.get('supported_performance_hints')}")
    for name, size in info["files"].items():
        print(f"   {name:40} {size:>15,} bytes")
    if not info["sections"]:
        print("   weights.bin: no readable container (opaque from the first byte)")
        return
    total = info["files"]["weights.bin"]
    print("   weights.bin sections:")
    for s in info["sections"]:
        hdr = "readable" if s["header_readable"] else "encrypted header"
        print(f"     {s['name']:28} {s['size']:>15,} bytes {100 * s['size'] / total:5.1f}%"
              f"  @ {s['offset']:,}  ({hdr})")


def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    dirs = args or find_models()
    if not dirs:
        sys.exit(f"No models found in {DATA}. Download one with gnano-download.")
    infos = [inspect(d) for d in dirs]
    if "--json" in sys.argv:
        json.dump(infos, sys.stdout, indent=2)
        print()
    else:
        for info in infos:
            print_report(info)


if __name__ == "__main__":
    main()
