#ifndef OP_SPLIT_UTILS__H
#define OP_SPLIT_UTILS__H

#include <cassert>
#include <cstdint>
#include <inttypes.h>
#include <string>
#include <vector>
#include <mlir/IR/Operation.h>
#include "Helper/OpUtils.h"
#include "Helper/TensorUtils.h"
#include "Utils/mlirUtils.h"
#include "Helper/XPMLIRUtils.h"
#include "mlir/IR/Value.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"
#include "Utils/modulePrinter.h"

/// split_target: 只有被标记为split_target的算子才会进行split event推导
/// is_labeled: 0 表示算子不需要split, 1 表示算子需要split
/// is_sliced:  0 表示算子的输出没有处理, 1表示算子的输出已经被slice/concat
/// is_splited:  0 表示算子没有被split过, 1表示算子已经被split过了 

#define UNUSED(x) (void)(x)

namespace xp_mlir {
namespace xrt {

inline int64_t getSplitAlignValue() {
    // 所有算子需要保持一致, 否则split后形状不统一, 无法相连
    static int64_t align_value = 1;
    return align_value;
}

struct SplitEvent {
    int64_t do_split; // -1: unknown, 0: do not split, 1: do split
    int64_t splitDim;
    int64_t splitNum;
    SplitEvent() : do_split(-1), splitDim(-1), splitNum(-1) {}
    void setSplitOpInfo(int64_t dim, int64_t num) {
        do_split = 1;
        assert(dim >= 0 && num > 0 && "splitDim must be non-negative and splitNum must be positive");
        splitDim = dim;
        splitNum = num;
    }
    void setNoSplitInfo() {
        assert(this->isValid() == false && "event should be invalid before setting no split info");
        do_split = 0;
        splitDim = -1;
        splitNum = -1;
    }
    // if splitNum is 1, it means no split, so we consider it as invalid event
    bool isValid() const { return do_split == 1 && splitDim >= 0 && splitNum > 1; }
    bool isNosplit() const { return (do_split == 0); }
    bool isUnknown() const { return do_split == -1; }

    bool operator==(const SplitEvent& other) const {
        if (this->isValid() && other.isValid()) {
            if (splitDim == other.splitDim && splitNum == other.splitNum) return true;
        }

        if (this->isNosplit() && other.isNosplit()) return true;

        return false;
    }
    bool operator!=(const SplitEvent& other) const {
        return !(*this == other);
    }
    std::string to_string() const {
        return "SplitEvent(do_split=" + std::to_string(do_split) + ", splitDim=" + std::to_string(splitDim) + ", splitNum=" + std::to_string(splitNum) + ")";
    }
};

class DefaultEventOp {
public:
    static SplitEvent SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events);
    static std::vector<SplitEvent> SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event);
};

class MatmulEventOp {
public:
    static SplitEvent SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events);
    static std::vector<SplitEvent> SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event);
};

class EletwEventOp {
public:
    static SplitEvent SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events);
    static std::vector<SplitEvent> SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event);
};

class QuantEventOp {
public:
    static SplitEvent SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events);
    static std::vector<SplitEvent> SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event);
};

class TransposeEventOp {
public:
    static SplitEvent SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events);
    static std::vector<SplitEvent> SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event);
};

class ReshapeEventOp {
public:
    static SplitEvent SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events);
    static std::vector<SplitEvent> SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event);
};

class ReduceEventOp {
public:
    static SplitEvent SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events);
    static std::vector<SplitEvent> SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event);
};

class FullConnectEventOp {
public:
    static SplitEvent SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events);
    static std::vector<SplitEvent> SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event);
};

mlir::DictionaryAttr event2DictionaryAttr(mlir::MLIRContext* ctx, const SplitEvent& event);
SplitEvent dictionaryAttr2Event(mlir::DictionaryAttr splitInfo);
void setTensorSplitAttribute(mlir::Value val, const SplitEvent& event, mlir::Operation* owner);
SplitEvent getTensorSplitEvent(mlir::Value val);
std::string getOpSignature(mlir::Operation* op);
mlir::ModuleOp getModuleFromValue(mlir::Value val);
int64_t transposeInputDim2OutputDim(const std::vector<int64_t>& perm, int64_t input_dim);
int64_t transposeOutputDim2InputDim(const std::vector<int64_t>& perm, int64_t output_dim);
int64_t indexInContainer(int64_t value, const std::vector<int64_t>& container);
int64_t getOperandIndex(mlir::Operation* op, mlir::Value operand);
mlir::Operation* getAliasOp(mlir::Value val);
mlir::Operation* createAliasOpAfterValue(mlir::Value val, const SplitEvent& event);
std::vector<SplitEvent> getOpInputEvents(mlir::Operation* op);
std::vector<int64_t> splitDim(const SplitEvent& event, const std::vector<int64_t>& original_shape, int64_t align_value, std::vector<std::vector<int64_t>>& split_shapes);
void createSplitValueSliceAndConcat(mlir::Value val, mlir::PatternRewriter &rewriter);
mlir::Operation* createSliceOp(const mlir::Value &val, mlir::PatternRewriter &rewriter,
                                const std::vector<int64_t> &outShape, std::vector<int64_t> axes, 
                                    std::vector<int64_t> starts, std::vector<int64_t> ends, std::vector<int64_t> steps, int64_t idx_slice);
void splitNoRegionOp(mlir::Operation* op, mlir::PatternRewriter &rewriter);
bool isOpWithRegion(Operation *op);
bool isOpInRegionOp(Operation *op);
void splitRegionOp(mlir::Operation* op, mlir::PatternRewriter &rewriter);
void eraseValueSplitInfo(mlir::Value val);
void setOpLabeled(mlir::Operation* op, int64_t type);
bool isOpLabeled(mlir::Operation* op);
bool isOpInfered(mlir::Operation* op);
void unsetOpLabeled(mlir::Operation* op);
void setOpSliced(mlir::Operation* op);
void unsetOpSliced(mlir::Operation* op);
bool isOpSliced(mlir::Operation* op);
void setOpSplited(mlir::Operation* op);
void unsetOpSplited(mlir::Operation* op);
bool isOpSplited(mlir::Operation* op);
void setOpSplitIndex(mlir::Operation* op, int64_t split_index);
void unsetOpSplitIndex(mlir::Operation* op);
bool ifOpHasSplitIndex(mlir::Operation* op);
int64_t getOpSplitIndex(mlir::Operation* op);
void unsetModuleAllOpsAttr(mlir::ModuleOp module, const std::string& attr_name);
void eraseModuleAllValuesSplitInfo(mlir::ModuleOp module);

void setSplitTarget(mlir::Operation* op);
bool isSplitTarget(mlir::Operation* op);

void setInsertedSliceConcat(mlir::Operation* op);
bool isInsertedSliceConcat(mlir::Operation* op);

void setOpMoved(mlir::Operation* op);
bool isOpMoved(mlir::Operation* op);

template<typename T, template <typename> class Container>
bool is_in_container(const Container<T>& vec, const T& value) {
    return vec.end() != std::find(vec.begin(), vec.end(), value);
}

enum class AxisDep : int64_t {
    IS_ONE = 0,
    A_FULL_B_PARTIAL = 1,
    EQUAL,
    A_PARTIAL_B_PARTIAL,
    A_PARTIAL_B_FULL
};

struct AxesRelation {
    std::vector<std::vector<int64_t>> dep_axes; // for each input axis, the related output axes
    std::vector<std::vector<AxisDep>> dep_types; // for each input axis, the dependency type with related output axes

    std::string to_string() const;
};
AxesRelation reshapeInputOutputAxisRelations(const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape);
void unittest_reshapeInputOutputAxisRelations();
int64_t getReshapeInferAxis(const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape, const SplitEvent& input_event);

void foldConstantSlice(mlir::Operation *slice_op, mlir::PatternRewriter &rewriter);

void dumpDebugMlirFile(mlir::Operation* op);
void dumpDebugMlirFile(mlir::Value val);

void custom_assert(bool flag, const std::string& message, mlir::Operation* op);

bool isBlockArgument(mlir::Value val);
std::vector<mlir::Value> getOpDependentValues(mlir::Operation* op);
mlir::Value getLatestDefinedValueInModule(const std::vector<Value> &values);

int32_t getSplitUserNum(Operation *op);

} // namespace xrt
} // namespace xp_mlir


#endif