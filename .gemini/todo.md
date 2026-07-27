# Danh sách công việc (TODO List)

- [x] Tích hợp WandB (Weights & Biases) cho các mô hình.
- [x] Cấu hình đường dẫn thư mục gốc động `ROOT_DIR`.
- [x] Đồng bộ phân chia tập dữ liệu thống nhất cho tất cả các mô hình:
  - **Train**: 80% (0 $\rightarrow$ 80%)
  - **Val**: 10% ở giữa (80% $\rightarrow$ 90%)
  - **Test**: 10% ở cuối (90% $\rightarrow$ 100%)
- [x] Các tệp đã đồng bộ phân chia dữ liệu:
  - `stgcn.py`
  - `hybrid.py`
  - `stgcn_block_attn.py`
  - `stgcn_mixed_blocks.py`
  - `compare_models.py`
- [x] Xuất báo cáo tài liệu kiến trúc ra `architecture_summary.md`.
