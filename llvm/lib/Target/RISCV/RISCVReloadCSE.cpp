//===- RISCVReloadCSE.cpp - CSE for spill slot reloads -------------------===//
//
// Eliminates redundant VL1R/VL1RE8 reloads from the same spill slot
// when they are close together and the slot hasn't been rewritten.
// Runs after greedy register allocation, before VirtRegRewriter.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "riscv-reload-cse"

static cl::opt<int> ReloadCSEGapThreshold(
    "riscv-reload-cse-gap",
    cl::desc("Max gap between reloads to merge (slot index units)"),
    cl::init(256), cl::Hidden);

namespace {

class RISCVReloadCSE : public MachineFunctionPass {
public:
  static char ID;
  RISCVReloadCSE() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "RISCV Reload CSE"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace

char RISCVReloadCSE::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVReloadCSE, "riscv-reload-cse",
                      "RISCV Reload CSE", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_END(RISCVReloadCSE, "riscv-reload-cse",
                    "RISCV Reload CSE", false, false)

static bool hasSlotStoreBetween(int Slot, SlotIndex From, SlotIndex To,
                                MachineFunction &MF, LiveIntervals &LIS) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr()) continue;
      SlotIndex SI = LIS.getInstructionIndex(MI);
      if (SI < From) continue;
      if (SI > To) break;
      unsigned Opc = MI.getOpcode();
      if (Opc == RISCV::VS1R_V || Opc == RISCV::VS2R_V ||
          Opc == RISCV::VS4R_V || Opc == RISCV::VS8R_V) {
        for (MachineOperand &MO : MI.operands())
          if (MO.isFI() && MO.getIndex() == Slot) return true;
      }
      if (MI.isCall()) return true;
    }
  }
  return false;
}

static void mergeReloads(
    SmallVectorImpl<std::pair<SlotIndex, Register>> &Reloads,
    int SpillSlot, LiveIntervals &LIS, MachineFunction &MF) {
  if (Reloads.size() < 2) return;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register FirstVReg = Reloads[0].second;

  SlotIndex MergedStart = Reloads[0].first;
  SlotIndex MergedEnd = LIS.getInterval(FirstVReg).endIndex();
  for (size_t i = 1; i < Reloads.size(); ++i) {
    const LiveInterval &LI = LIS.getInterval(Reloads[i].second);
    if (LI.empty()) continue;
    SlotIndex End = LI.endIndex();
    if (End > MergedEnd) MergedEnd = End;
  }

  LiveInterval &FirstLI = LIS.getInterval(FirstVReg);
  FirstLI.segments.clear();
  VNInfo *VNI = FirstLI.getNextValue(MergedStart, LIS.getVNInfoAllocator());
  FirstLI.addSegment(LiveRange::Segment(MergedStart, MergedEnd, VNI));

  for (size_t i = 1; i < Reloads.size(); ++i) {
    Register OldVReg = Reloads[i].second;
    MRI.replaceRegWith(OldVReg, FirstVReg);
    LIS.removeInterval(OldVReg);
    SlotIndex OldDef = Reloads[i].first;
    MachineInstr *OldMI = LIS.getInstructionFromIndex(OldDef);
    if (OldMI) {
      unsigned Opc = OldMI->getOpcode();
      if (Opc == RISCV::VL1RE8_V || Opc == RISCV::VL1RE32_V || Opc == RISCV::VL2RE8_V || Opc == RISCV::VL4RE8_V || Opc == RISCV::VL8RE8_V) {
        LIS.RemoveMachineInstrFromMaps(*OldMI);
        OldMI->eraseFromParent();
      }
    }
  }
}

bool RISCVReloadCSE::runOnMachineFunction(MachineFunction &MF) {
  LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  bool Changed = false;

  DenseMap<int, SmallVector<std::pair<SlotIndex, Register>, 4>> SlotReloads;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();
      if (Opc != RISCV::VL1RE8_V && Opc != RISCV::VL1RE32_V && Opc != RISCV::VL2RE8_V && Opc != RISCV::VL4RE8_V && Opc != RISCV::VL8RE8_V) continue;
      int Slot = -1;
      for (MachineOperand &MO : MI.operands()) {
        if (MO.isFI()) { Slot = MO.getIndex(); break; }
      }
      if (Slot < 0) continue;
      Register VReg = MI.getOperand(0).getReg();
      if (!VReg.isVirtual()) continue;
      SlotIndex Def = LIS.getInstructionIndex(MI).getRegSlot();
      SlotReloads[Slot].push_back({Def, VReg});
    }
  }

  for (auto &Entry : SlotReloads) {
    int Slot = Entry.first;
    auto &Reloads = Entry.second;
    if (Reloads.size() < 2) continue;

    llvm::sort(Reloads, [](auto &A, auto &B) { return A.first < B.first; });

    bool CanMerge = true;
    for (size_t i = 0; i < Reloads.size() - 1; ++i) {
      const LiveInterval &LI = LIS.getInterval(Reloads[i].second);
      if (LI.empty()) continue;
      SlotIndex EndI = LI.endIndex();
      SlotIndex DefJ = Reloads[i + 1].first;
      if (DefJ < EndI) continue;
      if (DefJ.distance(EndI) > ReloadCSEGapThreshold) { CanMerge = false; break; }
    }
    if (!CanMerge) continue;

    SlotIndex FirstDef = Reloads[0].first;
    const LiveInterval &LastLI = LIS.getInterval(Reloads.back().second);
    if (LastLI.empty()) continue;
    if (hasSlotStoreBetween(Slot, FirstDef, LastLI.endIndex(), MF, LIS))
      continue;

    mergeReloads(Reloads, Slot, LIS, MF);
    Changed = true;
    LLVM_DEBUG(dbgs() << "RISCVReloadCSE: merged " << Reloads.size()
                      << " reloads from slot " << Slot << "\n");
  }
  return Changed;
}

FunctionPass *llvm::createRISCVReloadCSEPass() {
  return new RISCVReloadCSE();
}
