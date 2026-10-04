# Run Chrome's Gemini Nano without Chrome.
#
#   make                  build gnano, gnano-server and libnano_dawn.so in build/release/
#   make DEBUG=1          debug build (no optimization, for gdb) in build/debug/
#   sudo make install     install to /usr/local (or: make install PREFIX=~/.local)
#   make download         download the GPU model    (make download-cpu: CPU model)
#   make inspect          show the downloaded models' file layout
#   make verify           test the models against the public Gemma 3n E2B
#   make uninstall        remove an install          (same PREFIX as install)
#   make clean            remove build/
#   make distclean        also remove the downloaded tools in third_party/
#
# The first `make` downloads Chromium's clang and libc++ headers and builds Dawn,
# which takes a few minutes. Models live in ~/.local/share/gemini-nano.

PREFIX ?= /usr/local
BINDIR  = $(PREFIX)/bin
LIBDIR  = $(PREFIX)/lib/gemini-nano

# Pinned to Chrome 154.0.8037.57: Chromium's tools/clang/scripts/update.py
# (clang) and DEPS (libcxx_revision, dawn_revision).
CLANG_PKG  = clang-llvmorg-24-init-3796-g20e97c4b-27
LIBCXX_REV = 97b436da4c33663581d394f4ee0a5977fc38c2f4
DAWN_REV   = 8597a1aaec546a7f19e8e662a49ebf03b2e91308
# stb_image, for decoding images given to --image (github.com/nothings/stb).
STB_REV    = 2c980bb59875b0d32144a71867fbdebb2f77cd20

TP     = third_party
ifdef DEBUG
BUILD  = build/debug
OPT    = -O0 -g
else
BUILD  = build/release
OPT    = -O2 -g
endif
CXX    = $(TP)/clang/bin/clang++
LIBCXX = $(TP)/libcxx/include/string
STB    = $(TP)/stb/stb_image.h

# Chrome's library passes std:: objects across its API, so we compile with the
# same C++ library setup as Chrome: Chromium's libc++ (ABI v2, std::__Cr), with
# Chromium's __config_site. Only header-only parts of libc++ are used;
# operator new etc. come from libstdc++.
CXXFLAGS = -std=c++20 $(OPT) -Wall -Wextra -MMD -MP \
  -nostdinc++ -isystem chromium/libcxx-config -isystem $(TP)/libcxx/include \
  -Ichromium/include -isystem $(TP)/stb \
  -D_LIBCPP_DISABLE_EXTERN_TEMPLATE -fno-exceptions -fno-rtti
LDLIBS = -nostdlib++ -lstdc++ -ldl -lpthread

all: $(BUILD)/gnano $(BUILD)/gnano-server $(BUILD)/libnano_dawn.so

$(BUILD)/gnano: $(BUILD)/main.o $(BUILD)/engine.o $(BUILD)/image.o
	$(CXX) $^ -o $@ $(LDLIBS)

$(BUILD)/gnano-server: $(BUILD)/server.o $(BUILD)/engine.o $(BUILD)/image.o
	$(CXX) $^ -o $@ $(LDLIBS)

$(BUILD)/%.o: src/%.cc | $(CXX) $(LIBCXX) $(STB)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(wildcard $(BUILD)/*.d)

$(CXX):
	@echo "==> Downloading Chromium's clang (one-time)..."
	mkdir -p $(TP)/clang
	curl -fL "https://commondatastorage.googleapis.com/chromium-browser-clang/Linux_x64/$(CLANG_PKG).tar.xz" \
	  | tar xJ -C $(TP)/clang

$(LIBCXX):
	@echo "==> Downloading Chromium's libc++ headers (one-time)..."
	mkdir -p $(TP)/libcxx/include
	curl -fL "https://chromium.googlesource.com/external/github.com/llvm/llvm-project/libcxx/+archive/$(LIBCXX_REV)/include.tar.gz" \
	  | tar xz -C $(TP)/libcxx/include

$(STB):
	mkdir -p $(TP)/stb
	curl -fL -o $@ "https://raw.githubusercontent.com/nothings/stb/$(STB_REV)/stb_image.h"

# Dawn (WebGPU) for the GPU backend. Chrome compiles it into the chrome binary,
# so we build the same revision and expose its proc table (src/dawn/). It is
# always built optimized, once, and copied into each build folder.
$(BUILD)/libnano_dawn.so: src/dawn/shim.cc src/dawn/CMakeLists.txt | $(CXX) $(TP)/dawn/CMakeLists.txt
	@echo "==> Building Dawn (one-time, a few minutes)..."
	cmake -S src/dawn -B $(TP)/dawn-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
	  -DDAWN_SRC="$(CURDIR)/$(TP)/dawn" \
	  -DCMAKE_C_COMPILER="$(CURDIR)/$(TP)/clang/bin/clang" \
	  -DCMAKE_CXX_COMPILER="$(CURDIR)/$(CXX)" >/dev/null
	ninja -C $(TP)/dawn-build nano_dawn
	@mkdir -p $(BUILD)
	cp $(TP)/dawn-build/libnano_dawn.so $@

$(TP)/dawn/CMakeLists.txt:
	@echo "==> Downloading Dawn (~1.3 GB with dependencies, one-time)..."
	rm -rf $(TP)/dawn
	git init -q $(TP)/dawn
	git -C $(TP)/dawn fetch -q --depth 1 https://dawn.googlesource.com/dawn $(DAWN_REV)
	git -C $(TP)/dawn checkout -q FETCH_HEAD
	cd $(TP)/dawn && python3 tools/fetch_dawn_dependencies.py -s

# Installs the release build. gnano finds libnano_dawn.so next to its real
# (symlink-resolved) location.
install: BUILD = build/release
install:
	@test -x $(BUILD)/gnano -a -x $(BUILD)/gnano-server -a -f $(BUILD)/libnano_dawn.so \
	  || { echo "Nothing built yet: run 'make' first (without sudo)."; exit 1; }
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(BINDIR)
	install -m 755 $(BUILD)/gnano $(BUILD)/gnano-server scripts/download-model.py $(DESTDIR)$(LIBDIR)/
	install -m 644 $(BUILD)/libnano_dawn.so $(DESTDIR)$(LIBDIR)/
	ln -sf ../lib/gemini-nano/gnano $(DESTDIR)$(BINDIR)/gnano
	ln -sf ../lib/gemini-nano/gnano-server $(DESTDIR)$(BINDIR)/gnano-server
	ln -sf ../lib/gemini-nano/download-model.py $(DESTDIR)$(BINDIR)/gnano-download
	@echo "==> Installed gnano, gnano-server and gnano-download in $(BINDIR)"

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/gnano $(DESTDIR)$(BINDIR)/gnano-server $(DESTDIR)$(BINDIR)/gnano-download
	rm -rf $(DESTDIR)$(LIBDIR)
	@echo "==> Uninstalled. Models are kept in ~/.local/share/gemini-nano."

download:
	python3 scripts/download-model.py

download-cpu:
	python3 scripts/download-model.py --cpu

inspect:
	python3 scripts/inspect-model.py

verify: $(BUILD)/gnano $(BUILD)/libnano_dawn.so
	python3 scripts/verify-model.py --gnano $(BUILD)/gnano

clean:
	rm -rf build

distclean: clean
	rm -rf $(TP)

.PHONY: all install uninstall download download-cpu inspect verify clean distclean
