#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

namespace {
  struct CmpPass : public ModulePass {
    static char ID;
    LLVMContext *C;
    const DataLayout *DL;
    CmpPass() : ModulePass(ID) {}

    bool runOnModule(Module& M) override {
      C = &(M.getContext());
      DL = &M.getDataLayout();
      AttributeList SanCovTraceCmpZeroExtAL;
      Type *VoidTy = Type::getVoidTy(*C);
      IRBuilder<> IRB(*C);

      Type* Int8Ty = IRB.getInt8Ty();
      Type* Int16Ty = IRB.getInt16Ty();
      Type* Int64Ty = IRB.getInt64Ty();

      FunctionCallee CallbackFunc = M.getOrInsertFunction(
        "__sn_cmp",
        SanCovTraceCmpZeroExtAL,
        VoidTy,
        Int8Ty,
        Int16Ty,
        Int64Ty,
        Int64Ty 
      );

      uint32_t cmpIdx = 0;

      for (auto &F: M) {
        for (auto &BB: F) {
          for (auto &Inst: BB) {
            if (CmpInst * ICMP = dyn_cast<CmpInst>(&Inst)) {
              Instruction *nextInst = Inst.getNextNode();
              if (BranchInst *BR = dyn_cast<BranchInst>(nextInst)) {
                if (BR->isConditional()) {
                  IRBuilder<> IRB(ICMP);
                  Value *A0 = ICMP->getOperand(0);
                  Value *A1 = ICMP->getOperand(1);
                  if (!A0->getType()->isIntegerTy()) continue;
                  uint8_t TypeSize = DL->getTypeStoreSizeInBits(A0->getType());
                  if (TypeSize > 64) continue;
                  bool FirstIsConst = isa<ConstantInt>(A0);
                  bool SecondIsConst = isa<ConstantInt>(A1);
                  if (FirstIsConst && SecondIsConst) continue;
                  if (FirstIsConst || SecondIsConst) {
                    if (SecondIsConst) std::swap(A0, A1);
                    auto Ty = Type::getIntNTy(*C, TypeSize);
                    IRB.CreateCall(CallbackFunc, {
                      ConstantInt::get(Ty, TypeSize),
                      ConstantInt::get(Ty, cmpIdx),
                      IRB.CreateIntCast(A0, Ty, true),
                      IRB.CreateIntCast(A1, Ty, true)
                    });
                    cmpIdx = cmpIdx + 1;
                  }
                }
              }
            }
          }
        }
      }
      outs() << "cmpIdx: " << cmpIdx << "\n";

      return false;
    }
  };
}

char CmpPass::ID = 0;

static void registerCmpPass(
  const PassManagerBuilder &,
  legacy::PassManagerBase &PM
) {
  PM.add(new CmpPass());
}
static RegisterStandardPasses RegisterCmpPass(
  PassManagerBuilder::EP_ModuleOptimizerEarly, registerCmpPass);

static RegisterStandardPasses RegisterCmpPass0(
  PassManagerBuilder::EP_EnabledOnOptLevel0, registerCmpPass);
