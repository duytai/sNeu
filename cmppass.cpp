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
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Dominators.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>

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

namespace {
  struct CmpPass : public ModulePass {
    static char ID;
    LLVMContext *C;
    const DataLayout *DL;

    CmpPass() : ModulePass(ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
			ModulePass::getAnalysisUsage(AU);
			AU.addRequired<DominatorTreeWrapperPass>();
		}

    virtual StringRef getPassName() const override {
			return "Pass To Collect Branch Distances";
		}

    std::string getFunctionSignature(Function* F)  {
      std::stringstream Str;
      Str << F->getName().data() << ":" << F->arg_size();
      return Str.str();
    }

    u64 createUniqId(StringRef str) {
      u64 UniqId = 0;
      llvm::MD5 md5;
      llvm::MD5::MD5Result Result;
      md5.update(str);
      md5.final(Result);
      for (auto I = 0; I < 8; ++ I)
        UniqId |= static_cast<u64>(Result[I]) << (I * 8);
      return UniqId;
    }

    bool runOnModule(Module& M) override {

      u64 From = createUniqId(M.getModuleIdentifier());

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

      for (auto &F: M) {
        if (!F.empty()) {
          std::string FSig = getFunctionSignature(&F);
          DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
          std::vector<std::pair<DomTreeNode*, u64>> Stack;
          auto Item = std::pair<DomTreeNode*, u64>(DT.getRootNode(), From);
          Stack.push_back(Item);
          while (Stack.size()) {
            auto Node = Stack.back().first;
            auto Parent = Stack.back().second;
            Stack.pop_back();
            BasicBlock *BB = Node->getBlock();
            for (auto &Inst: *BB) {
              if (CallInst* CALL = dyn_cast<CallInst>(&Inst)) {
                Function* CalledFunction = CALL->getCalledFunction();
                if (CalledFunction) {
                  if (!CalledFunction->isIntrinsic()) {
                    auto CSig = getFunctionSignature(CalledFunction);
                    std::stringstream VarName;
                    VarName << "__sn_(" << FSig << ")_(" << CSig << ")_" << From;
                    new GlobalVariable(
                      M,
                      Int64Ty,
                      false,
                      GlobalValue::CommonLinkage,
                      ConstantInt::get(Int64Ty, 0),
                      VarName.str()
                    );
                  }
                }
              }
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
                        ConstantInt::get(Int64Ty, From + 1),
                        IRB.CreateIntCast(A0, Ty, true),
                        IRB.CreateIntCast(A1, Ty, true)
                      });
                      std::stringstream VarName;
                      VarName << "__sn_(" << FSig << ")_" << Parent << "_" << From + 1 ;
                      new GlobalVariable(
                        M,
                        Int64Ty,
                        false,
                        GlobalValue::CommonLinkage,
                        ConstantInt::get(Int64Ty, 0),
                        VarName.str()
                      );
                      From = From + 1;
                    }
                  }
                }
              }
            }
            for (auto Child: *Node) {
              auto Item = std::pair<DomTreeNode*, u64>(Child, From);
              Stack.push_back(Item);
            }
          }
        }
      }
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
