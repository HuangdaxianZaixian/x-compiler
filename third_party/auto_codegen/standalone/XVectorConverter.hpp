#ifndef XVECTOR_CONVERTER__HPP
#define XVECTOR_CONVERTER__HPP

#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "standalone/StandalonePattern.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "xp_mlir/Dialect/XMA/Vector/XVectorOps.hpp"
#include <cstdint>
#include <memory>

namespace xp_mlir {
namespace xp_std {

struct ScfHandler {
    ScfHandler() : new_standalone_op(nullptr), scf_region(nullptr), outer_loop(nullptr), inner_loop(nullptr),
                   outer_loop_index(nullptr), inner_loop_index(nullptr) {}

    mlir::Operation* new_standalone_op;
    mlir::Region* scf_region;
    mlir::scf::ForOp outer_loop;
    mlir::scf::ForOp inner_loop;
    mlir::Value outer_loop_index;
    mlir::Value inner_loop_index;
    std::vector<int64_t> iter_shape;
    std::vector<mlir::Value> iter_indices;
    std::vector<mlir::Operation*> loop_index_increment_ops;
};

class XVectorConverter {
public:
    XVectorConverter(std::shared_ptr<StandalonePattern> pattern) : standalone_pattern_(pattern) {}
    void convert();

    std::shared_ptr<StandalonePattern> getStandalonePattern() { return standalone_pattern_; }
    ScfHandler& getScfHandler() { return scf_handler_; }
    void destoryNewStandaloneOp();

private:
    std::vector<int64_t> getIterationShape();
    std::vector<int64_t> getVectorLoopBoundaries(const std::vector<int64_t> shape);
    void xmaOptoXvectorOp(mlir::Operation* op, const StandaloneOpInfo& op_info);
    void convertToScfLoops();
    void convertToXvector();
    void addStandaloneL1Store();
    void addLoopConfig();
    void addRvvConfig();
    xvec::InitXregOp allocXregForScalar(mlir::OpBuilder& builder, mlir::Location loc, int32_t scalar, int32_t xreg_id, int32_t scalar_bitwidth);
    xvec::ScalarRegType getOpOutputScalarRegType(mlir::OpBuilder& builder, int32_t xreg_id, int32_t bitwidth);
    xvec::VectorRegType getOpOutputVectorRegType(mlir::OpBuilder& builder, int32_t vreg_id, int32_t bitwidth);
    mlir::Value getGeneralL1InputIterVreg(mlir::OpBuilder& builder, mlir::Location loc, const OpOperandInfo& operand_info);
    mlir::Value getQuantParamL1InputIterVreg(mlir::OpBuilder& builder, mlir::Location loc, const OpOperandInfo& operand_info);
    mlir::Value getGeneralL1OutputIterAddr(mlir::OpBuilder& builder, mlir::Location loc, const OpOperandInfo& operand_info);
    mlir::Operation* buildXvectorOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Operation* op, const std::vector<mlir::Value>& input_values, xvec::VectorRegType output_type);

private:
    std::shared_ptr<StandalonePattern> standalone_pattern_;
    ScfHandler scf_handler_;
    std::map<int32_t, mlir::Value> id_2_vreg_value_table_;
};

} // namespace xp_std
} // namespace xp_mlir

#endif // XVECTOR_CONVERTER__HPP