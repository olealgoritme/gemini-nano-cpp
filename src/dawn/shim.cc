// Exposes Dawn's native proc table, as Chrome does in chrome_ml.cc:
//   chrome_ml->TryInitDawnProcs(dawn::native::GetProcs());
#include <dawn/dawn_proc_table.h>
#include <dawn/native/DawnNative.h>

extern "C" __attribute__((visibility("default"))) const DawnProcTable*
NanoGetDawnProcs() {
  return &dawn::native::GetProcs();
}

extern "C" __attribute__((visibility("default"))) unsigned NanoDawnProcTableSize() {
  return sizeof(DawnProcTable);
}
