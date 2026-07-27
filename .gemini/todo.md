# Danh sách công việc (TODO List)

- [x] Tích hợp WandB (Weights & Biases) cho các mô hình.
- [x] Cấu hình đường dẫn thư mục gốc động `ROOT_DIR`.
- [x] Hoán đổi vị trí tập dữ liệu **Validation** (10% cuối) và **Test** (10% giữa).
- [x] **[MỚI]** Tạo tệp mô hình kết hợp `stgcn_mixed_blocks.py`:
  - **Block 1 & Block 2**: Khối STGCN nguyên bản (GLU 1 $\rightarrow$ Spatial ChebNet $\rightarrow$ GLU 2).
  - **Block 3**: Khối STGCN Attention (Attn 1 $\rightarrow$ Spatial ChebNet $\rightarrow$ Attn 2).
  - **Final Temporal Attention**: Đặt ở cuối mô hình trước khi chiếu ra Horizon.
- [x] Cập nhật tài liệu hướng dẫn (`walkthrough.md`).
