# Danh sách công việc (TODO List)

- [x] Tích hợp WandB (Weights & Biases) để log metric cho từng mô hình khi chạy riêng lẻ.
- [x] Tích hợp WandB vào `compare_models.py`.
- [x] Cấu hình đường dẫn thư mục gốc động `ROOT_DIR` cho tất cả các mô hình.
- [x] **[MỚI]** Cấu hình `hybrid.py` và `stgcn_block_attn.py`:
  - Đổi vị trí chia dữ liệu Validation và Test:
    - **Train**: 80% đầu tiên (0 $\rightarrow$ 80%)
    - **Test**: 10% ở giữa (80% $\rightarrow$ 90%)
    - **Val**: 10% cuối cùng (90% $\rightarrow$ 100%)
  - Cấu hình 2 Blocks (`NUM_BLOCKS = 2`).
- [x] Xuất báo cáo tài liệu kiến trúc ra `architecture_summary.md`.
