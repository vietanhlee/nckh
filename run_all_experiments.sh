#!/usr/bin/env bash
# ==============================================================================
# 🚀 AUTOMATED EXPERIMENT RUNNER SCRIPT (IEEE PAPER BENCHMARK SUITE)
# ==============================================================================
# Script này tự động thực thi 4 phần thử nghiệm độc lập của bài báo:
#   1. benchmark_5seeds.py              : Stage 2 Graph Forecasting Benchmark (15 Models, 5 Seeds)
#   2. run_noise_robustness_study.py    : Noise Robustness Study (Heteroscedastic Perception Noise)
#   3. train_and_visualize_attention.py : Temporal Attention Matrix Interpretability Heatmaps
#   4. train_counting.py                : Stage 1 Vision Perception Benchmark & Grad-CAM++
# ==============================================================================

# Thoát ngay nếu có lỗi xảy ra
set -e

# Đặt màu hiển thị trên Terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Đường dẫn mặc định tới thư mục dữ liệu
DATA_ROOT="${1:-/workspace/GRAPH}"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${GREEN}🚀 BẮT ĐẦU CHẠY TOÀN BỘ CHUỖI THỰC NGHIỆM NCKH (IEEE PAPER BENCHMARK SUITE)${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo -e "📅 Thời gian khởi chạy : $(date)"
echo -e "📂 Thư mục dữ liệu     : ${DATA_ROOT}"
echo -e "🖥️ GPU Device           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU Mode')"
echo -e "${BLUE}==============================================================================${NC}"

# Tạo các thư mục lưu kết quả nếu chưa tồn tại
mkdir -p logs plots paper/fig checkpoints

# ------------------------------------------------------------------------------
# 📍 THỬ NGHIỆM 1: STAGE 2 GRAPH FORECASTING 5-SEEDS BENCHMARK (15 MODELS)
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/4] 📈 Đang chạy Stage 2: Graph Forecasting Benchmark 15 Models (5 Seeds) ...${NC}"
python benchmark_5seeds.py --root_dir "${DATA_ROOT}"

echo -e "${GREEN}✅ [1/4] Hoàn thành Stage 2! Báo cáo đã lưu tại benchmark_5seeds_report.md, JSON kết quả và paper/fig/${NC}"

# ------------------------------------------------------------------------------
# 📍 THỬ NGHIỆM 2: NOISE ROBUSTNESS STUDY (PERCEPTION NOISE RESILIENCE)
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/4] 🛡️ Đang chạy Noise Robustness Study (Thử nghiệm chịu nhiễu nhận dạng) ...${NC}"
python run_noise_robustness_study.py --root_dir "${DATA_ROOT}"

echo -e "${GREEN}✅ [2/4] Hoàn thành Noise Robustness Study! Biểu đồ đã lưu vào plots/ và paper/fig/${NC}"

# ------------------------------------------------------------------------------
# 📍 THỬ NGHIỆM 3: TEMPORAL ATTENTION WEIGHT INTERPRETABILITY & HEATMAPS
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/4] 🧠 Đang chạy Temporal Attention Interpretability & Heatmap Analysis ...${NC}"
python train_and_visualize_attention.py --root_dir "${DATA_ROOT}" --epochs 80

echo -e "${GREEN}✅ [3/4] Hoàn thành Temporal Attention Analysis! Ma trận Attention đã lưu vào paper/fig/${NC}"

# ------------------------------------------------------------------------------
# 📍 THỬ NGHIỆM 4: STAGE 1 VISION PERCEPTION BENCHMARK (COUNTING & GRAD-CAM++)
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/4] 📸 Đang chạy Stage 1: Vision Perception Benchmark & Grad-CAM++ ...${NC}"
python train_counting.py

echo -e "${GREEN}✅ [4/4] Hoàn thành Stage 1! Báo cáo đã lưu tại counting_benchmark_report.md và paper/fig/${NC}"

# ------------------------------------------------------------------------------
# 🏁 KẾT THÚC CHUỖI THỰC NGHIỆM
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}🎉 TẤT CẢ 4 PHẦN THỰC NGHIỆM ĐÃ HOÀN THÀNH XUẤT SẮC!${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo -e "📊 Báo cáo thống kê   : benchmark_5seeds_report.md & counting_benchmark_report.md"
echo -e "💾 Dữ liệu JSON thô    : benchmark_5seeds_results.json"
echo -e "🖼️ Biểu đồ bài báo     : paper/fig/ & plots/"
echo -e "💾 Trọng số Checkpoints : checkpoints/"
echo -e "📝 File Nhật ký (Logs)  : logs/"
echo -e "${BLUE}==============================================================================${NC}"
