#include "standalone/XVectorConverter.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "standalone/RegManager.hpp"
#include "Helper/GraphUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Helper/XPMLIRUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "standalone/RegManager.hpp"
#include "standalone/StandaloneOpInfo.hpp"
#include "xp_mlir/Dialect/XMA/IR/XMAOps.h"
#include "xp_mlir/Dialect/XMA/Vector/XVectorOps.hpp"
#include "llvm/IR/Type.h"
#include <cstdint>
#include "mlir/IR/Verifier.h"

namespace xp_mlir {
namespace xp_std {
std::vector<int64_t> XVectorConverter::getIterationShape() {
    auto standalone_region_op = standalone_pattern_->getStandaloneRegionOp();
    auto all_op_infos = standalone_pattern_->getOp2StandaloneInfoTable();
    auto standalone_region_op_info = all_op_infos.at(standalone_region_op);

    auto shape = standalone_region_op_info.output_info.shape;
    assert(shape.size() == 4 && "the shape size is expected to be no more than 4");
    return shape;
}

std::vector<int64_t> XVectorConverter::getVectorLoopBoundaries(const std::vector<int64_t> shape) {
    assert(shape.size() >= 2 && "the shape size is expected to be no less than 2");
    auto c = shape.back();
    auto n = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<int64_t>());
    const int32_t vector_lane_num = VEU_VECTOR_LANE_NUM;
    return {n, (c + vector_lane_num - 1) / vector_lane_num};
}

void XVectorConverter::convert() {
    // 创建带scf loop的新region
    convertToScfLoops();

    // 将xma op转换成xvector op
    convertToXvector();

    // 添加standalone region的输出l1 store指令
    addStandaloneL1Store();

    // 添加loop begin/end op
    addLoopConfig();

    // 添加rvv配置指令
    addRvvConfig();

    // 检查ir是否合法
    if (failed(mlir::verify(scf_handler_.new_standalone_op))) {
        llvm::errs() << "ir verification failed!\n";
        std::abort();
    }

    scf_handler_.new_standalone_op->dump();

    // 后处理
    // auto old_standalone_region_op = standalone_pattern_->getStandaloneRegionOp();
    // old_standalone_region_op->replaceAllUsesWith(scf_handler_.new_standalone_op);
    // old_standalone_region_op->erase();
}

void XVectorConverter::destoryNewStandaloneOp() {
    if (scf_handler_.new_standalone_op) {
        scf_handler_.new_standalone_op->erase();
        scf_handler_.new_standalone_op = nullptr;
    }
}

xvec::InitXregOp XVectorConverter::allocXregForScalar(mlir::OpBuilder& builder, mlir::Location loc, int32_t scalar, int32_t xreg_id, int32_t scalar_bitwidth) {
    // scalar -> xreg
    auto scalar_attr = IntegerAttr::get(builder.getIntegerType(32), scalar);
    auto scalar_data_type = xvec::symbolizeRvvDtype(scalar_bitwidth).value();
    auto output_reg_type = xvec::ScalarRegType::get(builder.getContext(), xreg_id, scalar_data_type);
    return builder.create<xvec::InitXregOp>(loc, output_reg_type, scalar_attr);
}

xvec::ScalarRegType XVectorConverter::getOpOutputScalarRegType(mlir::OpBuilder& builder, int32_t xreg_id, int32_t bitwidth) {
    auto data_type = xvec::symbolizeRvvDtype(bitwidth).value();
    auto output_reg_type = xvec::ScalarRegType::get(builder.getContext(), xreg_id, data_type);
    return output_reg_type;
}

xvec::VectorRegType XVectorConverter::getOpOutputVectorRegType(mlir::OpBuilder& builder, int32_t vreg_id, int32_t bitwidth) {
    auto data_type = xvec::symbolizeRvvDtype(bitwidth).value();
    auto output_reg_type = xvec::VectorRegType::get(builder.getContext(), vreg_id, data_type);
    return output_reg_type;
}

mlir::Value XVectorConverter::getGeneralL1InputIterVreg(mlir::OpBuilder& builder, mlir::Location loc, const OpOperandInfo& operand_info) {
    auto shape = operand_info.shape;
    assert(operand_info.mem_type == OpOperandMemType::L1 && "the operand memory type is expected to be l1");
    auto base_addr = operand_info.mem_value.l1_addr;
    assert(0 == base_addr % 128 && "the base address is expected to be 128-byte aligned");

    auto bitwidth = operand_info.bit_width;
    auto align_elem_size = getXmaBufferAlignValue(bitwidth);

    auto n_index_xreg = scf_handler_.iter_indices.at(0);
    auto h_index_xreg = scf_handler_.iter_indices.at(1);
    auto w_index_xreg = scf_handler_.iter_indices.at(2);
    auto c_index_xreg = scf_handler_.iter_indices.at(3);

    // auto outer_loop_index_increment_op = scf_handler_.loop_index_increment_ops.at(0);
    // auto inner_loop_index_increment_op = scf_handler_.loop_index_increment_ops.at(1);

    // auto N = shape.at(0);
    auto H = shape.at(1);
    auto W = shape.at(2);
    auto C = shape.at(3);
    
    auto align_C = align_up(C, align_elem_size);
    auto align_C_byte = align_C * ((bitwidth + 7) / 8);
    auto stride_c = VEU_VECTOR_LANE_NUM * ((bitwidth + 7) / 8);
    auto stride_w = align_C_byte;
    auto stride_h = W * stride_w;
    auto stride_n = H * stride_h;

    std::vector<int8_t> operand_broadcast_status(shape.size(), 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        // 如果某个维度是1, 则该维度index只能取0
        if (shape[i] == 1) {
            operand_broadcast_status[i] = 1;
        }
    }

    builder.setInsertionPoint(scf_handler_.inner_loop);
    auto outer_loop_base_addr_xreg = allocXregForScalar(builder, loc, base_addr, RegManager::get().allocScalarReg().id, 32);
    mlir::Operation* addr_acc_op = outer_loop_base_addr_xreg;
    // n offset
    if (0 == operand_broadcast_status.at(0)) {
        auto n_stride_xreg = allocXregForScalar(builder, loc, stride_n, RegManager::get().allocScalarReg().id, 32);
        auto n_offset_xreg = builder.create<xvec::Xmul>(loc, n_stride_xreg.getOutput().getType(), n_index_xreg, n_stride_xreg.getOutput());
        addr_acc_op = builder.create<xvec::Xadd>(loc, outer_loop_base_addr_xreg.getOutput().getType(), addr_acc_op->getResult(0), n_offset_xreg.getOutput());
    }

    // h offset
    if (0 == operand_broadcast_status.at(1)) {
        auto h_stride_xreg = allocXregForScalar(builder, loc, stride_h, RegManager::get().allocScalarReg().id, 32);
        auto h_offset_xreg = builder.create<xvec::Xmul>(loc, h_stride_xreg.getOutput().getType(), h_index_xreg, h_stride_xreg.getOutput());
        addr_acc_op = builder.create<xvec::Xadd>(loc, outer_loop_base_addr_xreg.getOutput().getType(), addr_acc_op->getResult(0), h_offset_xreg.getOutput());
    }

    // w offset
    if (0 == operand_broadcast_status.at(2)) {
        auto w_stride_xreg = allocXregForScalar(builder, loc, stride_w, RegManager::get().allocScalarReg().id, 32);
        auto w_offset_xreg = builder.create<xvec::Xmul>(loc, w_stride_xreg.getOutput().getType(), w_index_xreg, w_stride_xreg.getOutput());
        addr_acc_op = builder.create<xvec::Xadd>(loc, outer_loop_base_addr_xreg.getOutput().getType(), addr_acc_op->getResult(0), w_offset_xreg.getOutput());
    }

    builder.setInsertionPoint(scf_handler_.inner_loop.getBody()->getTerminator());
    // c offset
    if (0 == operand_broadcast_status.at(3)) {
        auto c_stride_xreg = allocXregForScalar(builder, loc, stride_c, RegManager::get().allocScalarReg().id, 32);
        auto c_offset_xreg = builder.create<xvec::Xmul>(loc, c_stride_xreg.getOutput().getType(), c_index_xreg, c_stride_xreg.getOutput());
        addr_acc_op = builder.create<xvec::Xadd>(loc, c_offset_xreg.getOutput().getType(), addr_acc_op->getResult(0), c_offset_xreg.getOutput());
    }

    auto addr_xreg = addr_acc_op->getResult(0);
    auto load_output_vreg_type = getOpOutputVectorRegType(builder, RegManager::get().allocVectorReg().id, bitwidth);
    auto l1_load_op = builder.create<xvec::VectorLoadOp>(loc, load_output_vreg_type, addr_xreg);
    
    return l1_load_op.getOutput();
}

mlir::Value XVectorConverter::getQuantParamL1InputIterVreg(mlir::OpBuilder& builder, mlir::Location loc, const OpOperandInfo& operand_info) {
    assert(operand_info.operand_type == OpOperandType::QUANT_PARAM && "the operand type is expected to be quant param");
    auto shape = operand_info.shape;
    assert(operand_info.mem_type == OpOperandMemType::L1 && "the operand memory type is expected to be l1");
    auto base_addr = operand_info.mem_value.l1_addr;
    assert(0 == base_addr % 128 && "the base address is expected to be 128-byte aligned");

    auto bitwidth = operand_info.bit_width;

    auto n_index_xreg = scf_handler_.iter_indices.at(0);
    auto h_index_xreg = scf_handler_.iter_indices.at(1);
    auto w_index_xreg = scf_handler_.iter_indices.at(2);
    auto c_index_xreg = scf_handler_.iter_indices.at(3);

    // auto outer_loop_index_increment_op = scf_handler_.loop_index_increment_ops.at(0);
    // auto inner_loop_index_increment_op = scf_handler_.loop_index_increment_ops.at(1);
    
    auto channel_dim = operand_info.quant_channel_dim;
    assert(channel_dim < 0 && "the quant channel dimension is expected to be less than 0, which means counting from the back");
    auto quant_channel_dim = shape.size() + channel_dim;
    assert(shape.back() == scf_handler_.iter_shape.at(quant_channel_dim) && 
                "the quant channel dimension size is expected to be the same as the corresponding iteration dimension size");

    builder.setInsertionPoint(scf_handler_.outer_loop);

    bool to_inner_loop = true;
    auto target_index_xreg = c_index_xreg;
    switch (quant_channel_dim) {
        case 0: {
            to_inner_loop = false;
            target_index_xreg = n_index_xreg;
            break;
        }
        case 1: {
            to_inner_loop = false;
            target_index_xreg = h_index_xreg;
            break;
        }
        case 2: {
            to_inner_loop = false;
            target_index_xreg = w_index_xreg;
            break;
        }
        case 3: {
            to_inner_loop = true;
            target_index_xreg = c_index_xreg;
            break;
        }
        default: {
            assert(false && "unsupported quant channel dimension");
            break;
        }
    }

    if (to_inner_loop) {
        builder.setInsertionPoint(scf_handler_.inner_loop.getBody()->getTerminator());
    } else {
        builder.setInsertionPoint(scf_handler_.inner_loop);
    }

    auto quant_param_base_addr_xreg = allocXregForScalar(builder, loc, base_addr, RegManager::get().allocScalarReg().id, 32);
    if (to_inner_loop) {
        // 量化轴为c轴
        auto stride_c = VEU_VECTOR_LANE_NUM * ((bitwidth + 7) / 8);
        auto c_stride_xreg = allocXregForScalar(builder, loc, stride_c, RegManager::get().allocScalarReg().id, 32);
        auto c_offset_xreg = builder.create<xvec::Xmul>(loc, c_stride_xreg.getOutput().getType(), c_index_xreg, c_stride_xreg.getOutput());
        auto final_param_addr = builder.create<xvec::Xadd>(loc, quant_param_base_addr_xreg.getOutput().getType(), quant_param_base_addr_xreg.getOutput(), c_offset_xreg.getOutput());

        auto addr_xreg = final_param_addr.getOutput();
        auto load_output_vreg_type = getOpOutputVectorRegType(builder, RegManager::get().allocVectorReg().id, bitwidth);
        auto l1_load_op = builder.create<xvec::VectorLoadOp>(loc, load_output_vreg_type, addr_xreg);
        return l1_load_op.getOutput();
    } else {
        // 根据迭代index计算当前加载的是第几笔vector数据, 以及量化参数在vector的index
        auto veu_lane_num_xreg = allocXregForScalar(builder, loc, VEU_VECTOR_LANE_NUM, RegManager::get().allocScalarReg().id, 32);
        auto num_stride_output_type = getOpOutputScalarRegType(builder, RegManager::get().allocScalarReg().id, 32);
        auto num_stride = builder.create<xvec::Xdiv>(loc, num_stride_output_type, target_index_xreg, veu_lane_num_xreg.getOutput());
        auto quant_param_index_output_type = getOpOutputScalarRegType(builder, RegManager::get().allocScalarReg().id, 32);
        auto quant_param_index = builder.create<xvec::Xmod>(loc, quant_param_index_output_type, target_index_xreg, num_stride.getOutput(), veu_lane_num_xreg.getOutput());
        auto stride_c = VEU_VECTOR_LANE_NUM * ((bitwidth + 7) / 8);
        auto c_stride_xreg = allocXregForScalar(builder, loc, stride_c, RegManager::get().allocScalarReg().id, 32);
        auto c_offset_xreg = builder.create<xvec::Xmul>(loc, c_stride_xreg.getOutput().getType(), num_stride.getOutput(), c_stride_xreg.getOutput());
        auto final_param_addr = builder.create<xvec::Xadd>(loc, quant_param_base_addr_xreg.getOutput().getType(), quant_param_base_addr_xreg.getOutput(), c_offset_xreg.getOutput());

        auto addr_xreg = final_param_addr.getOutput();
        auto load_output_vreg_type = getOpOutputVectorRegType(builder, RegManager::get().allocVectorReg().id, bitwidth);
        auto l1_load_op = builder.create<xvec::VectorLoadOp>(loc, load_output_vreg_type, addr_xreg);

        auto select_scalar_to_vreg_output_type = getOpOutputVectorRegType(builder, RegManager::get().allocVectorReg().id, 32);
        auto select_op = builder.create<xvec::SelectScalarFromVectorAndBroadcast>(loc, select_scalar_to_vreg_output_type, l1_load_op.getOutput(), quant_param_index.getOutput());
        return select_op.getOutput();
    }

    return {};
}

mlir::Value XVectorConverter::getGeneralL1OutputIterAddr(mlir::OpBuilder& builder, mlir::Location loc, const OpOperandInfo& operand_info) {
    auto shape = operand_info.shape;
    assert(operand_info.mem_type == OpOperandMemType::L1 && "the operand memory type is expected to be l1");
    auto base_addr = operand_info.mem_value.l1_addr;
    assert(0 == base_addr % 128 && "the base address is expected to be 128-byte aligned");

    auto bitwidth = operand_info.bit_width;
    auto align_elem_size = getXmaBufferAlignValue(bitwidth);
    
    auto C = shape.at(3);
    auto align_C = align_up(C, align_elem_size);
    auto align_C_byte = align_C * ((bitwidth + 7) / 8);
    auto stride_c = VEU_VECTOR_LANE_NUM * ((bitwidth + 7) / 8);

    auto outer_loop_index = scf_handler_.outer_loop_index;
    auto inner_loop_index = scf_handler_.inner_loop_index;

    builder.setInsertionPoint(scf_handler_.inner_loop);
    auto outer_loop_base_addr_xreg = allocXregForScalar(builder, loc, base_addr, RegManager::get().allocScalarReg().id, 32);
    // outter offset
    auto outer_stride_xreg = allocXregForScalar(builder, loc, align_C_byte, RegManager::get().allocScalarReg().id, 32);
    auto outer_offset_xreg = builder.create<xvec::Xmul>(loc, outer_stride_xreg.getOutput().getType(), outer_loop_index, outer_stride_xreg.getOutput());
    builder.create<xvec::Xadd>(loc, outer_loop_base_addr_xreg.getOutput().getType(), outer_loop_base_addr_xreg.getOutput(), outer_offset_xreg.getOutput());
    

    builder.setInsertionPoint(scf_handler_.inner_loop.getBody()->getTerminator());
    auto addr_xreg = outer_loop_base_addr_xreg.getOutput();
    
    auto c_stride_xreg = allocXregForScalar(builder, loc, stride_c, RegManager::get().allocScalarReg().id, 32);
    auto c_offset_xreg = builder.create<xvec::Xmul>(loc, c_stride_xreg.getOutput().getType(), inner_loop_index, c_stride_xreg.getOutput());
    auto final_addr_op = builder.create<xvec::Xadd>(loc, c_offset_xreg.getOutput().getType(), outer_loop_base_addr_xreg.getOutput(), c_offset_xreg.getOutput());
    addr_xreg = final_addr_op.getOutput();

    return addr_xreg;
}

void XVectorConverter::convertToScfLoops() {
    mlir::Operation* op = standalone_pattern_->getStandaloneRegionOp();
    auto callOp = llvm::dyn_cast<xma::CallOp>(op);
    if (!callOp) return;

    // 生成新的old, 但不删除原op, 也不更新连接关系
    auto new_op = cloneOpAndAppendNewRegion(callOp);
    assert(new_op && "failed to clone op with new region");
  
    auto ctx = new_op->getContext();
    auto block = &(new_op->getRegion(1).getBlocks().front());
    mlir::OpBuilder builder(ctx);
    auto loc = op->getLoc();
    builder.setInsertionPointToEnd(block);

    // 插入两层loops
    auto shape = getIterationShape();
    auto loop_bounds = getVectorLoopBoundaries(shape);
    assert(loop_bounds.size() == 2 && "the loop boundaries size is expected to be 2");

    auto loop_start = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto loop_step = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto loop_bound_0 = builder.create<mlir::arith::ConstantIndexOp>(loc, loop_bounds[0]);
    auto loop_bound_1 = builder.create<mlir::arith::ConstantIndexOp>(loc, loop_bounds[1]);

    auto outer_loop = builder.create<mlir::scf::ForOp>(loc, loop_start, loop_bound_0, loop_step);
    builder.setInsertionPoint(outer_loop.getBody()->getTerminator());
    auto inner_loop = builder.create<mlir::scf::ForOp>(loc, loop_start, loop_bound_1, loop_step);

    // 添加迭代index
    // loop0-index -> xreg, ++loop-index
    builder.setInsertionPoint(outer_loop);
    auto outer_loop_index_xreg = RegManager::get().allocScalarReg();
    auto outer_loop_index_xreg_op = allocXregForScalar(builder, loc, -1, outer_loop_index_xreg.id, 32);
    builder.setInsertionPointToStart(outer_loop.getBody());
    auto outer_loop_index_increment_op =builder.create<xvec::Xincrement>(loc, outer_loop_index_xreg_op.getOutput().getType(), outer_loop_index_xreg_op.getOutput());
    (void)(outer_loop_index_increment_op);

    // loop1-index -> xreg, ++loop-index
    builder.setInsertionPoint(inner_loop);
    auto inner_loop_index_xreg = RegManager::get().allocScalarReg();
    auto inner_loop_index_xreg_op = allocXregForScalar(builder, loc, -1, inner_loop_index_xreg.id, 32);
    builder.setInsertionPointToStart(inner_loop.getBody());
    auto inner_loop_index_increment_op =builder.create<xvec::Xincrement>(loc, inner_loop_index_xreg_op.getOutput().getType(), inner_loop_index_xreg_op.getOutput());
    (void)(inner_loop_index_increment_op);

    // 插入迭代变量计算
    assert(shape.size() == 4 && "the shape size is expected to be no more than 4");
    // const int64_t N = shape[0];
    const int64_t H = shape[1];
    const int64_t W = shape[2];
    // const int64_t C = shape[3];

    builder.setInsertionPoint(outer_loop);
    auto HW_xreg_op = allocXregForScalar(builder, loc, H*W, RegManager::get().allocScalarReg().id, 32);
    auto W_xreg_op = allocXregForScalar(builder, loc, W, RegManager::get().allocScalarReg().id, 32);

    builder.setInsertionPointAfter(outer_loop_index_increment_op);
    auto n_index_output_type = getOpOutputScalarRegType(builder, RegManager::get().allocScalarReg().id, 32);
    auto n_index = builder.create<xvec::Xdiv>(loc, n_index_output_type, outer_loop_index_increment_op.getOutput(), HW_xreg_op.getOutput());

    auto mod_HW_output_type = getOpOutputScalarRegType(builder, RegManager::get().allocScalarReg().id, 32);
    auto mod_HW = builder.create<xvec::Xmod>(loc, mod_HW_output_type, outer_loop_index_increment_op.getOutput(), n_index.getOutput(), HW_xreg_op.getOutput());

    auto h_index_output_type = getOpOutputScalarRegType(builder, RegManager::get().allocScalarReg().id, 32);
    auto h_index = builder.create<xvec::Xdiv>(loc, h_index_output_type, mod_HW.getOutput(), W_xreg_op.getOutput());

    auto w_index_output_type = getOpOutputScalarRegType(builder, RegManager::get().allocScalarReg().id, 32);
    auto w_index = builder.create<xvec::Xmod>(loc, w_index_output_type, mod_HW.getOutput(), h_index.getOutput(), W_xreg_op.getOutput());
    (void)(n_index);
    (void)(h_index);
    (void)(w_index);

    scf_handler_.new_standalone_op = new_op;
    scf_handler_.scf_region = &(new_op->getRegion(1));
    scf_handler_.outer_loop = outer_loop;
    scf_handler_.inner_loop = inner_loop;
    scf_handler_.outer_loop_index = outer_loop_index_increment_op.getOutput();
    scf_handler_.inner_loop_index = inner_loop_index_increment_op.getOutput();
    scf_handler_.iter_shape = shape;
    scf_handler_.iter_indices = {n_index.getOutput(), h_index.getOutput(), w_index.getOutput(), inner_loop_index_increment_op.getOutput()};
    scf_handler_.loop_index_increment_ops = {outer_loop_index_increment_op, inner_loop_index_increment_op};
}

void XVectorConverter::convertToXvector() {
    auto all_ops = standalone_pattern_->getAllStandaloneRegionOps();
    auto all_op_infos = standalone_pattern_->getOp2StandaloneInfoTable();
    for (auto op : all_ops) {
        auto op_info = all_op_infos.at(op);
        xmaOptoXvectorOp(op, op_info);
    }
}

void XVectorConverter::addStandaloneL1Store() {
    auto all_ops = standalone_pattern_->getAllStandaloneRegionOps();
    auto last_op_in_region = all_ops.back();
    auto old_standalone_region_op = standalone_pattern_->getStandaloneRegionOp();
    auto all_op_infos = standalone_pattern_->getOp2StandaloneInfoTable();
    auto standalone_region_op_info = all_op_infos.at(old_standalone_region_op);
    auto last_op_info = all_op_infos.at(last_op_in_region);

    assert(last_op_info.output_info.mem_type == OpOperandMemType::V_REG && "the output of the last op in standalone region is expected to be stored in vreg");
    auto output_vreg_id = last_op_info.output_info.mem_value.v_reg_id;
    assert(id_2_vreg_value_table_.count(output_vreg_id) && "vreg id is expected to be found in the table");
    auto output_vreg_value = id_2_vreg_value_table_[output_vreg_id];

    assert(standalone_region_op_info.output_info.mem_type == OpOperandMemType::L1 && "the output of standalone region op must be stored to l1");

    auto ctx = last_op_in_region->getContext();
    mlir::OpBuilder builder(ctx);
    auto loc = last_op_in_region->getLoc();
    builder.setInsertionPoint(scf_handler_.inner_loop.getBody()->getTerminator());

    // l1-addr -> xreg
    auto iter_l1_addr = getGeneralL1OutputIterAddr(builder, loc, standalone_region_op_info.output_info);

    // store op
    auto l1_store_op = builder.create<xvec::StoreOp>(loc, output_vreg_value, iter_l1_addr);
    (void)(l1_store_op);
}

void XVectorConverter::addLoopConfig() {
    auto ctx = scf_handler_.scf_region->getContext();
    mlir::OpBuilder builder(ctx);
    auto loc = scf_handler_.scf_region->getLoc();

    std::vector<mlir::scf::ForOp> loop_scf_op = {scf_handler_.outer_loop, scf_handler_.inner_loop};
    std::vector<int32_t> loop_ids = {0, 1};

    for (size_t i = 0; i < loop_scf_op.size(); ++i) {
        auto scf_for_op = loop_scf_op.at(i);

        // begin
        builder.setInsertionPointToStart(scf_for_op.getBody());

        auto upper_bound_value = scf_for_op.getUpperBound();
        auto upper_bound_def_op = upper_bound_value.getDefiningOp<mlir::arith::ConstantIndexOp>();
        assert(upper_bound_def_op && "the loop upper bound is expected to be defined by constant index op");
        auto upper_bound = upper_bound_def_op.value();

        auto loop_time_xreg_id = (i == 0) ? VEU_OUTER_LOOP_TIMES_XREG_ID : VEU_INNER_LOOP_TIMES_XREG_ID;

        auto loop_times_xreg_type = getOpOutputScalarRegType(builder, RegManager::get().allocSpecificScalarReg(loop_time_xreg_id).id, 32);
        auto loop_times_xreg = builder.create<xvec::InitXregOp>(loc, loop_times_xreg_type, upper_bound);
        // 移动到prelogue中
        loop_times_xreg->moveBefore(scf_handler_.outer_loop);
        auto loop_id_attr = mlir::IntegerAttr::get(builder.getIntegerType(32), loop_ids.at(i));
        auto loop_begin_op = builder.create<xvec::LoopBegin>(loc, loop_times_xreg.getOutput(), loop_id_attr);
        (void)(loop_begin_op);

        // end
        builder.setInsertionPoint(scf_for_op.getBody()->getTerminator());
        auto loop_end_op = builder.create<xvec::LoopEnd>(loc, loop_id_attr);
        (void)(loop_end_op);
    }
}

void XVectorConverter::addRvvConfig() {
    auto ctx = scf_handler_.scf_region->getContext();
    mlir::OpBuilder builder(ctx);
    auto loc = scf_handler_.scf_region->getLoc();

    builder.setInsertionPoint(scf_handler_.outer_loop);

    auto vl_set_xreg = RegManager::get().allocSpecificScalarReg(VEU_RVV_CONFIG_VL_SET_XREG_ID);
    auto vl_return_xreg = RegManager::get().allocSpecificScalarReg(VEU_RVV_CONFIG_VL_RETURN_XREG_ID);

    auto rvv_vl_set_xreg_type = getOpOutputScalarRegType(builder, vl_set_xreg.id, 32);
    auto rvv_vl_set_xreg = builder.create<xvec::InitXregOp>(loc, rvv_vl_set_xreg_type, VEU_VECTOR_LANE_NUM);
    auto rvv_vl_return_xreg_type = getOpOutputScalarRegType(builder, vl_return_xreg.id, 32);
    auto rvv_vl_config_op = builder.create<xvec::RvvConfig>(loc, rvv_vl_return_xreg_type, rvv_vl_set_xreg.getOutput());
    (void)(rvv_vl_config_op);
}

void XVectorConverter::xmaOptoXvectorOp(mlir::Operation* op, const StandaloneOpInfo& op_info) {
    // lut op of quant-lut
    auto quant_lut_partion = standalone_pattern_->getQuantLutPartition();
    if (quant_lut_partion.quant_op_ && op == quant_lut_partion.quant_op_) {
        // 信息合并到lut op中
        return;
    }

    auto ctx = op->getContext();
    mlir::OpBuilder builder(ctx);
    auto loc = op->getLoc();
    builder.setInsertionPoint(scf_handler_.inner_loop.getBody()->getTerminator());

    auto all_input_infos = op_info.input_infos;
    auto output_info = op_info.output_info;
 
    if (quant_lut_partion.lut_op_ && op == quant_lut_partion.lut_op_) {
        assert(quant_lut_partion.quant_op_ && "the quant op in quant-lut partition is expected to be not null");

        auto op_2_standalone_info_table = standalone_pattern_->getOp2StandaloneInfoTable();
        auto quant_input_op_infos = op_2_standalone_info_table.at(quant_lut_partion.quant_op_).input_infos;
        auto lut_input_op_infos = op_info.input_infos;

        // x, quant-param, lut-table
        all_input_infos = {quant_input_op_infos.at(0), quant_input_op_infos.at(1), lut_input_op_infos.at(1)};
    }

    std::vector<mlir::Value> input_values;
    for (size_t i = 0; i < all_input_infos.size(); ++i) {
        const auto& input_info = all_input_infos[i];
        if (input_info.mem_type == OpOperandMemType::V_REG) {
            auto vreg_id = input_info.mem_value.v_reg_id;
            assert(id_2_vreg_value_table_.count(vreg_id) && "vreg id is expected to be found in the table");
            input_values.push_back(id_2_vreg_value_table_[vreg_id]); 
        } else if (input_info.mem_type == OpOperandMemType::L1) {
            if (input_info.operand_type == OpOperandType::QUANT_PARAM) {
                auto quant_param_vreg = getQuantParamL1InputIterVreg(builder, loc, input_info);
                input_values.push_back(quant_param_vreg);
            } else {
                auto general_input_vreg = getGeneralL1InputIterVreg(builder, loc, input_info);
                input_values.push_back(general_input_vreg);
            }
        } else if (input_info.mem_type == OpOperandMemType::X_REG) {
            // scalar -> xreg
            auto scalar = input_info.mem_value.x_reg_scalar;
            auto scalar_bitwidth = input_info.bit_width;
            auto to_xreg_op = allocXregForScalar(builder, loc, scalar, RegManager::get().allocScalarReg().id, scalar_bitwidth);
            input_values.push_back(to_xreg_op.getOutput());
        } else if (input_info.mem_type == OpOperandMemType::IMM) {
            auto imm = input_info.mem_value.imm;
            auto imm_attr = IntegerAttr::get(builder.getIntegerType(32), imm);
            auto imm_output_type = xvec::RvvImmType::get(builder.getContext());
            auto imm_op = builder.create<xvec::ImmOp>(loc, imm_output_type, imm_attr);
            input_values.push_back(imm_op.getOutput());
        }
    }

    builder.setInsertionPoint(scf_handler_.inner_loop.getBody()->getTerminator());
    auto op_output_bitwidth = output_info.bit_width;
    assert(output_info.mem_type == OpOperandMemType::V_REG && "the output memory type is expected to be vreg");
    auto op_output_vreg_id = output_info.mem_value.v_reg_id;
    assert(RegManager::get().isUsedVectorReg(op_output_vreg_id) && "the output vreg id must be allocated");
    auto op_output_vreg_type = getOpOutputVectorRegType(builder, op_output_vreg_id, op_output_bitwidth);

    mlir::Operation* xvec_op = buildXvectorOp(builder, loc, op, input_values, op_output_vreg_type);

    // 注册输出
    // std::cout << "add vreg to table, vreg id: " << op_output_vreg_id << std::endl;
    id_2_vreg_value_table_[op_output_vreg_id] = xvec_op->getResult(0);
}


mlir::Operation* XVectorConverter::buildXvectorOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Operation* op, 
                                                  const std::vector<mlir::Value>& input_values, xvec::VectorRegType output_type) {
    // quant-lut
    auto quant_lut_partion = standalone_pattern_->getQuantLutPartition();
    if (quant_lut_partion.lut_op_ && op == quant_lut_partion.lut_op_) {
        // 需要将lut table参数移动到prelogue区域中
        auto lut_table_value = input_values.at(2);
        auto lut_table_def_op = lut_table_value.getDefiningOp();
        assert(lut_table_def_op && "the defining op of lut table value is expected to be not null");
        auto lut_table_xreg_init_op = llvm::dyn_cast<xvec::InitXregOp>(lut_table_def_op);
        assert(lut_table_xreg_init_op && "the defining op of lut table value is expected to be init-xreg op");
        lut_table_xreg_init_op->moveBefore(scf_handler_.outer_loop);

        auto vqlut_op = builder.create<xvec::Vqlut>(loc, output_type, input_values);

        auto op_2_standalone_info_table = standalone_pattern_->getOp2StandaloneInfoTable();
        auto quant_output_info = op_2_standalone_info_table.at(quant_lut_partion.quant_op_).output_info;
        auto quant_output_bitwidth = quant_output_info.bit_width;
        assert((quant_output_bitwidth == 8 || quant_output_bitwidth == 10) && "the quant output bitwidth is expected to be 8 or 10");
        int32_t table_size = (quant_output_bitwidth == 8) ? 256 : 1024;
        auto table_size_attr = builder.getIntegerAttr(builder.getIntegerType(32), table_size);

        vqlut_op.setTableSizeAttr(table_size_attr);

        return vqlut_op;
    }

    auto updated_input_values = input_values;
    if (llvm::isa<xma::AddOp, xma::SubOp, xma::DivOp>(op)) {
        // 添加extend op
        for (int32_t idx_input = 0; idx_input < input_values.size(); ++idx_input) {
            auto input_value = input_values.at(idx_input);
            if (!llvm::isa<xvec::VectorRegType>(input_value.getType())) {
                continue;
            }
            auto input_data_type = llvm::dyn_cast<xvec::VectorRegType>(input_value.getType()).getDtype();
            auto input_bitwidth = static_cast<int32_t>(input_data_type);
            
            if (input_bitwidth != 32) {
                auto extend_output_vreg = RegManager::get().allocVectorReg();
                auto extend_output_vreg_type = getOpOutputVectorRegType(builder, extend_output_vreg.id, 32);
                auto extend_op = builder.create<xvec::Extend>(loc, extend_output_vreg_type, input_value);
                updated_input_values.at(idx_input) = extend_op.getOutput();
            }
        }
    }
    
    mlir::Operation* xvec_op = llvm::TypeSwitch<mlir::Operation*, mlir::Operation*>(op)
        .Case<xma::ReciprocalOp>([&](auto) {
            return builder.create<xvec::ReciprocalOp>(loc, output_type, input_values);
        })
        .Case<xma::AddOp>([&](auto) {
            return builder.create<xvec::Vadd>(loc, output_type, updated_input_values);
        })
        .Case<xma::SubOp>([&](auto) {
            return builder.create<xvec::Vsub>(loc, output_type, updated_input_values);
        })
        .Case<xma::MulOp>([&](auto) {
            return builder.create<xvec::Vmul>(loc, output_type, input_values);
        })
        .Case<xma::DivOp>([&](auto) {
            return builder.create<xvec::Vdiv>(loc, output_type, updated_input_values);
        })
        .Case<xma::QuantOp, xma::DeQuantOp>([&](auto) {
            // 添加extend op
            auto input_data_type = llvm::dyn_cast<xvec::VectorRegType>(input_values.at(0).getType()).getDtype();
            auto input_bitwidth = static_cast<int32_t>(input_data_type);
            auto output_data_type = llvm::dyn_cast<xvec::VectorRegType>(output_type).getDtype();
            auto output_bitwidth = static_cast<int32_t>(output_data_type);

            auto updated_quant_input_values = input_values;
            if (output_bitwidth < 32 && input_bitwidth < 32) {
                auto extend_output_vreg = RegManager::get().allocVectorReg();
                auto extend_output_vreg_type = getOpOutputVectorRegType(builder, extend_output_vreg.id, 32);
                auto extend_op = builder.create<xvec::Extend>(loc, extend_output_vreg_type, input_values.at(0));
                updated_quant_input_values.at(0) = extend_op.getOutput();
            }
            return builder.create<xvec::Vquant>(loc, output_type, updated_quant_input_values);
        })
        .Default([&](auto) {
            assert(false && "unsupported op type");
            return nullptr;
        });

    return xvec_op;
}
    

} // namespace xp_std
} // namespace xp_mlir

