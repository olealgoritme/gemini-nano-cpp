#!/usr/bin/env python3
"""Download Chrome's Gemini Nano model into ~/.local/share/gemini-nano.

Chrome only downloads the model when a *person* clicks or types on a page that
asks for it (a "user gesture"). This script fakes that click: it starts a
hidden Chrome on a dedicated profile, opens a tiny local page, and sends a real
mouse click into it over Chrome's DevTools pipe. Then it waits for the ~4 GB
download to finish, showing progress.

It also copies Chrome's model library next to the models (lib/), so gnano
keeps working even if Chrome later updates itself.

Only needs python3 and Google Chrome. Usage (installed as gnano-download):
    download-model.py            # GPU model, into .../profile-gpu
    download-model.py --cpu      # CPU model, into .../profile-cpu
    download-model.py --show     # visible Chrome window (if hidden fails)
"""

import argparse
import glob
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time

# Per-user data folder shared with gnano/gnano-server (see DataDir() in
# src/engine.cc): $NANO_HOME, else $XDG_DATA_HOME/gemini-nano, else
# ~/.local/share/gemini-nano.
DATA = os.environ.get("NANO_HOME") or os.path.join(
    os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"),
    "gemini-nano")
PROFILE = os.path.join(DATA, "profile-gpu")  # set in main()
LIB_NAME = "liboptimization_guide_internal.so"
# The C++ headers in chromium/include/ match this Chrome version exactly.
BUILT_FOR = "154.0.8037.57"

PAGE = b"""<!doctype html>
<html style="height:100%"><body style="height:100%;margin:0">
<p>Downloading Gemini Nano... you can leave this alone.</p>
<script>
window.S = {state: 'init', progress: 0, error: null};
(async () => {
  if (!('LanguageModel' in self)) { S.state = 'no-api'; return; }
  S.state = await LanguageModel.availability();
})();
// Chrome requires a user gesture to start the download; the script clicks here.
document.addEventListener('click', async () => {
  try {
    S.state = 'starting';
    const m = await LanguageModel.create({monitor(mon) {
      mon.addEventListener('downloadprogress', e => {
        S.state = 'downloading';
        S.progress = e.loaded;
      });
    }});
    m.destroy();
    S.state = await LanguageModel.availability();
  } catch (e) {
    S.state = 'error';
    S.error = String(e);
  }
});
</script></body></html>"""


def die(msg):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_chrome():
    if os.environ.get("NANO_CHROME"):
        return os.environ["NANO_CHROME"]
    for name in ("google-chrome-stable", "google-chrome", "google-chrome-beta",
                 "google-chrome-unstable"):
        path = shutil.which(name)
        if path:
            return path
    die("Google Chrome not found. Install it from https://www.google.com/chrome/\n"
        "(Chromium does NOT work: it lacks Google's model library.)\n"
        "Or set NANO_CHROME=/path/to/google-chrome")


def find_model():
    """Newest downloaded model folder in PROFILE, or None."""
    hits = sorted(glob.glob(os.path.join(PROFILE, "OptGuideOnDeviceModel", "*", "weights.bin")))
    return os.path.dirname(hits[-1]) if hits else None


def serve_page():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


class DevTools:
    """Minimal Chrome DevTools Protocol client over --remote-debugging-pipe.

    Chrome reads commands from fd 3 and writes replies to fd 4, each message
    being JSON terminated by a NUL byte.
    """

    def __init__(self, chrome, args):
        to_chrome_r, self.to_chrome = os.pipe()
        self.from_chrome, from_chrome_w = os.pipe()

        def child_fds():
            os.dup2(to_chrome_r, 3)
            os.dup2(from_chrome_w, 4)

        self.proc = subprocess.Popen(
            [chrome, "--remote-debugging-pipe", *args],
            # subprocess closes stray fds *after* preexec_fn, so keep 3 and 4.
            pass_fds=(3, 4), preexec_fn=child_fds,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(to_chrome_r)
        os.close(from_chrome_w)
        self.buf = b""
        self.next_id = 0

    def call(self, method, session=None, **params):
        self.next_id += 1
        msg = {"id": self.next_id, "method": method, "params": params}
        if session:
            msg["sessionId"] = session
        os.write(self.to_chrome, json.dumps(msg).encode() + b"\0")
        while True:
            while b"\0" not in self.buf:
                chunk = os.read(self.from_chrome, 65536)
                if not chunk:
                    die("Chrome exited unexpectedly. Is another Chrome using "
                        f"{PROFILE}? Close it and retry.")
                self.buf += chunk
            raw, self.buf = self.buf.split(b"\0", 1)
            reply = json.loads(raw)
            if reply.get("id") == self.next_id:
                if "error" in reply:
                    die(f"DevTools {method}: {reply['error'].get('message')}")
                return reply.get("result", {})

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def copy_library(chrome):
    real = os.path.realpath(chrome)
    src = os.path.join(os.path.dirname(real), LIB_NAME)
    if not os.path.exists(src):
        src = os.path.join("/opt/google/chrome", LIB_NAME)
    if not os.path.exists(src):
        print(f"  (could not find {LIB_NAME} next to Chrome; gnano will look in /opt/google/chrome)")
        return
    os.makedirs(os.path.join(DATA, "lib"), exist_ok=True)
    shutil.copy2(src, os.path.join(DATA, "lib", LIB_NAME))
    print(f"  Copied {src} -> {os.path.join(DATA, 'lib', LIB_NAME)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    variant = ap.add_mutually_exclusive_group()
    variant.add_argument("--gpu", action="store_true",
                         help="download the GPU model (default)")
    variant.add_argument("--cpu", action="store_true",
                         help="download the CPU model")
    ap.add_argument("--show", action="store_true",
                    help="show the Chrome window instead of running hidden")
    opts = ap.parse_args()
    global PROFILE
    PROFILE = os.path.join(DATA, "profile-cpu" if opts.cpu else "profile-gpu")

    chrome = find_chrome()
    version = subprocess.run([chrome, "--version"], capture_output=True,
                             text=True).stdout.strip()
    print(f"Using {version} ({chrome})")
    if BUILT_FOR not in version:
        print(f"  WARNING: gnano was written for Chrome {BUILT_FOR}. Other "
              "versions may crash it (see README).")

    existing = find_model()
    if existing:
        print(f"Model already downloaded: {existing}")
        copy_library(chrome)
        return

    port = serve_page()
    args = [
        f"--user-data-dir={PROFILE}",
        "--no-first-run", "--no-default-browser-check",
        "--disable-renderer-backgrounding",
        "--disable-background-timer-throttling",
        # OnDeviceModelForceCpuBackend makes Chrome offer only the CPU
        # performance hint, so it picks the CPU model variant.
        "--enable-features=OptimizationGuideOnDeviceModel:BypassPerfRequirement/true,AIPromptAPI"
        + (",OnDeviceModelForceCpuBackend" if opts.cpu else ""),
        "about:blank",
    ]
    if not opts.show:
        args.insert(0, "--headless=new")
    print(f"Starting Chrome ({'visible' if opts.show else 'hidden'}), profile: {PROFILE}")
    dt = DevTools(chrome, args)
    try:
        target = dt.call("Target.createTarget", url=f"http://127.0.0.1:{port}/")["targetId"]
        session = dt.call("Target.attachToTarget", targetId=target, flatten=True)["sessionId"]

        def page_state():
            r = dt.call("Runtime.evaluate", session, expression="S", returnByValue=True)
            return r.get("result", {}).get("value") or {"state": "init"}

        # Wait for the page to report the initial availability.
        deadline = time.time() + 30
        state = page_state()
        while state["state"] == "init" and time.time() < deadline:
            time.sleep(0.5)
            state = page_state()
        print(f"Model availability: {state['state']}")

        if state["state"] == "no-api":
            die("This Chrome has no LanguageModel (Prompt API). Update Chrome.")
        if state["state"] == "unavailable":
            die("Chrome says this machine can't run Gemini Nano. It needs ~22 GB "
                "free disk, and >4 GB VRAM or 16 GB RAM + 4 CPU cores.\n"
                "Try --show, then open chrome://on-device-internals for details.")

        if state["state"] != "available":
            # The "user gesture": a real mouse click delivered to the page.
            for kind in ("mousePressed", "mouseReleased"):
                dt.call("Input.dispatchMouseEvent", session, type=kind,
                        x=100, y=100, button="left", clickCount=1)
            print("Clicked; download starting (about 4 GB, this can take a while)...")

            last_change, last_progress = time.time(), -1
            while True:
                state = page_state()
                if state["state"] == "available":
                    break
                if state["state"] == "error":
                    die(f"Chrome refused: {state['error']}\n"
                        "Try again with --show, then check chrome://on-device-internals.")
                p = state.get("progress", 0)
                if p != last_progress:
                    last_progress, last_change = p, time.time()
                bar = "#" * int(p * 40)
                sys.stdout.write(f"\r  [{bar:<40}] {p * 100:5.1f}%  ({state['state']})")
                sys.stdout.flush()
                if time.time() - last_change > 600:
                    die("No progress for 10 minutes. Check your network, or try --show.")
                time.sleep(1)
            print("\r  [" + "#" * 40 + "] 100.0%  done          ")
    finally:
        dt.close()

    model = find_model()
    if not model:
        die("Chrome reported success but no weights.bin was found in the profile.")
    print(f"Model ready: {model}")
    copy_library(chrome)
    backend = "cpu" if opts.cpu else "gpu"
    print(f"\nNext: NANO_BACKEND={backend} gnano write a haiku about Linux")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run the script again to continue the download.")
        sys.exit(130)
