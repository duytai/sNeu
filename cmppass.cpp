#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

using namespace llvm;

#define u64 uint64_t
#define u32 uint32_t
#define s32 int32_t
#define u8  uint8_t

#define cLGN "\x1b[1;92m"
#define cRST "\x1b[0m"
#define bSTOP "\x0f"
#define RESET_G1 "\x1b)B"
#define CURSOR_SHOW "\x1b[?25h"
#define cLRD "\x1b[1;91m"
#define cBRI "\x1b[1;97m"

#define SAYF(x...) printf(x)

#define FATAL(x...) do { \
  SAYF(bSTOP RESET_G1 CURSOR_SHOW cRST cLRD "\n[-] PROGRAM ABORT : " \
      cBRI x); \
  SAYF(cLRD "\n         Location : " cRST "%s(), %s:%u\n\n", \
      __FUNCTION__, __FILE__, __LINE__); \
  exit(1); \
} while (0)

#define FORMAT(_str...) ({ \
  char * _tmp; \
  s32 _len = snprintf(NULL, 0, _str); \
  if (_len < 0) FATAL("snprintf() failed"); \
  _tmp = (char*) malloc(_len + 1); \
  snprintf(_tmp, _len + 1, _str); \
  _tmp; \
})

namespace {
  struct CmpPass : public ModulePass {
    static char ID;
    LLVMContext *C;
    const DataLayout *DL;

    CmpPass() : ModulePass(ID) {}

    bool runOnModule(Module& M) override {

      u64 From = 0, Size = 0;

      C = &(M.getContext());
      DL = &M.getDataLayout();
      AttributeList SanCovTraceCmpZeroExtAL;
      Type *VoidTy = Type::getVoidTy(*C);
      IRBuilder<> IRB(*C);

      Type* Int8Ty = IRB.getInt8Ty();
      Type* Int64Ty = IRB.getInt64Ty();

      FunctionCallee CallbackFunc = M.getOrInsertFunction(
        "__sn_cmp",
        SanCovTraceCmpZeroExtAL,
        VoidTy,
        Int8Ty,
        Int64Ty,
        Int64Ty,
        Int64Ty 
      );

      llvm::MD5 md5;
      llvm::MD5::MD5Result Result;
      md5.update(M.getModuleIdentifier());
      md5.final(Result);
      for (auto I = 0; I < 8; ++ I)
        From |= static_cast<uint64_t>(Result[I]) << (I * 8);

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
                  u8 TypeSize = DL->getTypeStoreSizeInBits(A0->getType());
                  if (TypeSize > 64) continue;
                  bool FirstIsConst = isa<ConstantInt>(A0);
                  bool SecondIsConst = isa<ConstantInt>(A1);
                  if (FirstIsConst && SecondIsConst) continue;
                  if (FirstIsConst || SecondIsConst) {
                    if (SecondIsConst) std::swap(A0, A1);
                    auto Ty = Type::getIntNTy(*C, TypeSize);
                    IRB.CreateCall(CallbackFunc, {
                      ConstantInt::get(Int8Ty, TypeSize),
                      ConstantInt::get(Int64Ty, From + Size),
                      IRB.CreateIntCast(A0, Ty, true),
                      IRB.CreateIntCast(A1, Ty, true)
                    });
                    Size = Size + 1;
                  }
                }
              }
            }
          }
        }
      }

      /* readelf -s file to get boundary */
      auto VarName = FORMAT("__sn_%lu_%lu", From, Size);
      new GlobalVariable(
        M,
        Int64Ty,
        false,
        GlobalValue::CommonLinkage,
        ConstantInt::get(Int64Ty, 0),
        VarName
      );

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
