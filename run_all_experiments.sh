#!/usr/bin/env bash
# ==============================================================================
# 🚀 AUTOMATED EXPERIMENT RUNNER SCRIPT (IEEE PAPER BENCHMARK SUITE)
# ==============================================================================
# Script này tự động thực thi các phần thử nghiệm của bài báo:
#   1. benchmark_5seeds.py              : Stage 2 Graph Forecasting Benchmark
#   2. train_and_visualize_attention.py : Temporal Attention Matrix Interpretability Heatmaps
#   3. train_counting.py                : Stage 1 Vision Perception Benchmark & Grad-CAM++
# ==============================================================================
# 💡 CÁCH SỬ DỤNG:
#   - Chạy chính thức đầy đủ   : ./run_all_experiments.sh
#   - Chạy test thử nhanh (1 epoch): ./run_all_experiments.sh --test
#   - Tùy chỉnh đường dẫn data : ./run_all_experiments.sh /path/to/data --test
# ==============================================================================

# Thoát ngay nếu có lỗi xảy ra
set -e

# Đặt màu hiển thị trên Terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Xử lý tham số đầu vào (DATA_ROOT & TEST_MODE)
TEST_MODE=false
DATA_ROOT="/workspace/GRAPH"

for arg in "$@"; do
    case $arg in
        --test|-t|test|TEST|True|true|1)
            TEST_MODE=true
            ;;
        --root=*)
            DATA_ROOT="${arg#*=}"
            ;;
        *)
            if [[ "$arg" != -* ]]; then
                DATA_ROOT="$arg"
            fi
            ;;
    esac
done

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${GREEN}🚀 KHỞI ĐỘNG CHUỖI THỰC NGHIỆM NCKH (IEEE PAPER BENCHMARK SUITE)${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo -e "📅 Thời gian khởi chạy : $(date)"
echo -e "📂 Thư mục dữ liệu     : ${DATA_ROOT}"
echo -e "🖥️ GPU Device           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU Mode')"

if [ "$TEST_MODE" = true ]; then
    echo -e "🧪 Chế độ thực thi     : ${YELLOW}TEST FAST DRY-RUN (Epochs = 1, Seeds = 1)${NC}"
    BENCHMARK_ARGS="--epochs 1 --seeds 42"
    ATTENTION_ARGS="--epochs 1"
    COUNTING_ARGS="--epochs 1 --seeds 42"
else
    echo -e "🏆 Chế độ thực thi     : ${GREEN}FULL BENCHMARK SUITE (Tất cả Seeds & Epochs đầy đủ)${NC}"
    BENCHMARK_ARGS=""
    ATTENTION_ARGS="--epochs 80"
    COUNTING_ARGS=""
fi
echo -e "${BLUE}==============================================================================${NC}"

# Tạo các thư mục lưu kết quả nếu chưa tồn tại
mkdir -p logs plots paper/fig checkpoints

# ------------------------------------------------------------------------------
# 📍 THỬ NGHIỆM 1: STAGE 2 GRAPH FORECASTING BENCHMARK
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/4] 📈 Đang chạy Stage 2: Graph Forecasting Benchmark ...${NC}"
python benchmark_5seeds.py --root_dir "${DATA_ROOT}" ${BENCHMARK_ARGS}

echo -e "${GREEN}✅ [1/4] Hoàn thành Stage 2! Báo cáo đã lưu tại benchmark_5seeds_report.md, JSON kết quả và paper/fig/${NC}"

# ------------------------------------------------------------------------------
# 📍 THỬ NGHIỆM 2: TEMPORAL ATTENTION WEIGHT INTERPRETABILITY & HEATMAPS
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/4] 🧠 Đang chạy Temporal Attention Interpretability & Heatmap Analysis ...${NC}"
python train_and_visualize_attention.py --root_dir "${DATA_ROOT}" ${ATTENTION_ARGS}

echo -e "${GREEN}✅ [2/4] Hoàn thành Temporal Attention Analysis! Ma trận Attention đã lưu vào paper/fig/${NC}"

# ------------------------------------------------------------------------------
# 📍 THỬ NGHIỆM 3: STAGE 1 VISION PERCEPTION BENCHMARK (COUNTING & GRAD-CAM++)
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/4] 📸 Đang chạy Stage 1: Vision Perception Benchmark & Grad-CAM++ ...${NC}"
python train_counting.py ${COUNTING_ARGS}

echo -e "${GREEN}✅ [3/4] Hoàn thành Stage 1! Báo cáo đã lưu tại counting_benchmark_report.md và paper/fig/${NC}"

# ------------------------------------------------------------------------------
# 📍 THỬ NGHIỆM 4: STAGE 3 NOISE ROBUSTNESS STUDY
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/4] 🌪️ Đang chạy Stage 3: Noise Robustness Study (Phân tích độ bền vững với nhiễu) ...${NC}"
python run_noise_robustness_study.py --root_dir "${DATA_ROOT}" ${BENCHMARK_ARGS}

echo -e "${GREEN}✅ [4/4] Hoàn thành Stage 3! Kết quả đã được lưu tại noise_robustness_report.md và paper/fig/${NC}"

# ------------------------------------------------------------------------------
# 🏁 KẾT THÚC CHUỖI THỰC NGHIỆM
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}🎉 TẤT CẢ 4 PHẦN THỰC NGHIỆM ĐÃ HOÀN THÀNH XUẤT SẮC!${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo -e "📊 Báo cáo thống kê   : benchmark_5seeds_report.md, counting_benchmark_report.md & noise_robustness_report.md"
echo -e "💾 Dữ liệu JSON thô    : benchmark_5seeds_results.json"
echo -e "🖼️ Biểu đồ bài báo     : paper/fig/ & plots/"
echo -e "💾 Trọng số Checkpoints : checkpoints/"
echo -e "📝 File Nhật ký (Logs)  : logs/"
echo -e "${BLUE}==============================================================================${NC}"
