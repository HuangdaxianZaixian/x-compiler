#include "SplitPatterns.h"
#include "Helper/OpUtils.h"
#include "Helper/TensorUtils.h"
#include "Utils/mlirUtils.h"
#include "Helper/XPMLIRUtils.h"
#include "Dialect/XRT/Rewrites/RewriteUtils.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"

namespace xp_mlir {
namespace xrt {

void pattern_matmul_k6144_n1536(mlir::ModuleOp moduleOp) {
    moduleOp.walk([](mlir::Operation* op) -> mlir::WalkResult {
        if (llvm::isa<xfr::XFRMatMulOp>(op)) {
            auto wgts_shape = getTensorShape(op->getOperand(1));
            if (wgts_shape.size() == 2 && wgts_shape[0] == 6144 && wgts_shape[1] == 1536) {
                // set split event
                SplitEvent event;
                event.setSplitOpInfo(2, 12);
                setTensorSplitAttribute(op->getResult(0), event, op);

                // set split region
                setSplitTarget(op);
             }
         }
        return mlir::WalkResult::advance();
    });
}

void pattern_matmul_k1536_n6144(mlir::ModuleOp moduleOp) {
    moduleOp.walk([](mlir::Operation* op) -> mlir::WalkResult {
        if (llvm::isa<xfr::XFRMatMulOp>(op)) {
            auto wgts_shape = getTensorShape(op->getOperand(1));
            if (wgts_shape.size() == 2 && wgts_shape[0] == 1536 && wgts_shape[1] == 6144) {
                // set split event
                SplitEvent event;
                event.setSplitOpInfo(2, 12);
                setTensorSplitAttribute(op->getResult(0), event, op);

                // set split region
                setSplitTarget(op);
             }
         }
        return mlir::WalkResult::advance();
    });
}

void pattern_pertoken(mlir::ModuleOp moduleOp) {
    moduleOp.walk([](mlir::Operation* op) -> mlir::WalkResult {
        if (llvm::isa<xfr::XFRMatMulOp>(op)) {
            if (getXfrValueName(op->getResult(0)) == "tests_unit_tests_quantization_test_pertoken_patterns_line_80_mat_res_int32") {
                std::cout << "find matmul op: " << op->getName().getStringRef().str() << std::endl;
                SplitEvent event;
                event.setSplitOpInfo(2, 4);
                setTensorSplitAttribute(op->getResult(0), event, op);

                // set split region
                setSplitTarget(op);
            }
        }
        return mlir::WalkResult::advance();
    });
}

void pattern_softmax(mlir::ModuleOp moduleOp) {
    moduleOp.walk([](mlir::Operation* op) -> mlir::WalkResult {
        if (llvm::isa<xfr::XFRReduceMaxOp>(op)) {
            if (getXfrValueName(op->getResult(0)) == "n_reduce_max") {
                std::cout << "find reducemax op: " << op->getName().getStringRef().str() << std::endl;
                SplitEvent event;
                event.setSplitOpInfo(1, 4);
                setTensorSplitAttribute(op->getResult(0), event, op);

                // set split region
                setSplitTarget(op);
            }
        }
        return mlir::WalkResult::advance();
    });
}

void pattern_moe_0(mlir::ModuleOp moduleOp) {
    // moduleOp.walk([](mlir::Operation* op) -> mlir::WalkResult {
    //     if (llvm::isa<xfr::XFRMatMulOp>(op)) {
    //         if (getXfrValueName(op->getResult(0)) == "var_145") {
    //             std::cout << "find TMU FUSE op: " << op->getName().getStringRef().str() << std::endl;
    //             SplitEvent event;
    //             event.setSplitOpInfo(2, 4);
    //             setTensorSplitAttribute(op->getResult(0), event, op);

    //             // set split region
    //             // setSplitTarget(op);
    //         }
    //     }
    //     return mlir::WalkResult::advance();
    // });

    // moduleOp.walk([](mlir::Operation* op) -> mlir::WalkResult {
    //     if (llvm::isa<xfr::XFRAddOp>(op)) {
    //         if (getXfrValueName(op->getResult(0)) == "workspace_zhangyh36_xiaopeng_com_fm_deploy_xpilot_vision_ai_foundation_projects_turing_turing_model_layer_norm_dynamic_tanh_line_23_add_int_2") {
    //             std::cout << "find add op: " << op->getName().getStringRef().str() << std::endl;
    //             SplitEvent event;
    //             event.setSplitOpInfo(2, 4);
    //             setTensorSplitAttribute(op->getResult(0), event, op);

    //             // set split region
    //             // setSplitTarget(op);
    //         }
    //     }
    //     return mlir::WalkResult::advance();
    // });

    moduleOp.walk([](mlir::Operation* op) -> mlir::WalkResult {
        if (llvm::isa<xfr::XFRTmuFuseOp>(op)) {
            auto user_num = getValueUserNums(op->getResult(0));
            if (user_num == 2) {
                mlir::Operation* user_0;
                mlir::Operation* user_1;
                for (auto user : op->getResult(0).getUsers()) {
                    if (user && llvm::isa<xfr::XFRMatMulOp>(user)) {
                        auto user_parent_op = user->getParentOp();
                        if (user_parent_op && llvm::isa<xfr::XFRTmuFuseOp>(user_parent_op)) {
                            auto tmp_user = *user_parent_op->getResult(0).getUsers().begin();
                            if (llvm::isa<xfr::XFRMulOp>(tmp_user)) {
                                user_0 = user_parent_op;
                            } 
                            if (llvm::isa<xfr::XFRMatMulOp>(tmp_user)) {
                                user_1 = user_parent_op;
                            }  
                        } 
                    }
                }

                if (user_0 && user_1) {
                    auto tmp_user = *user_1->getResult(0).getUsers().begin();
                    if (llvm::isa<xfr::XFRMatMulOp>(tmp_user)) {
                        auto user_parent_op = tmp_user->getParentOp();
                        if (user_parent_op && llvm::isa<xfr::XFRTmuFuseOp>(user_parent_op)) {

                            auto tmp_tmp_user = *user_parent_op->getResult(0).getUsers().begin();
                            if (llvm::isa<xfr::XFRMatMulOp>(tmp_tmp_user)) {
                                std::cout << "find TMU FUSE op: " << op->getName().getStringRef().str() << std::endl;
                                SplitEvent event;
                                event.setSplitOpInfo(1, 2);
                                setTensorSplitAttribute(op->getResult(0), event, op);

                                // set split region
                                // setSplitTarget(op);
                            }
                        }
                    }
                }
            }
        }
        return mlir::WalkResult::advance();
    });
}


}
}