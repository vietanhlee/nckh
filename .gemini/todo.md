# Danh sách công việc (TODO List)

- [x] Tích hợp WandB (Weights & Biases) để log metric cho từng mô hình khi chạy riêng lẻ.
- [x] Tích hợp WandB vào `compare_models.py`.
- [x] Cấu hình đường dẫn thư mục gốc động `ROOT_DIR` cho tất cả các mô hình.
- [x] **[MỚI]** Tạo tệp mô hình mới `stgcn_block_attn.py`:
  - Nhúng **Multi-Head Temporal Self-Attention** trực tiếp vào từng khối `STGCNBlockAttn` (thay thế lớp GLU thứ 2).
  - Cấu trúc từng block: `Temporal Gated Conv (GLU 1)` -> `Spatial Graph Conv (ChebNet)` -> `Temporal Self-Attention`.
  - Tăng capacity: `BLOCK_HIDDEN=64`, `NUM_BLOCKS=3`, `CHEB_K=3`, `DROPOUT=0.25`, `ATTN_NUM_HEADS=4`.
- [x] Tạo tài liệu hướng dẫn và báo cáo cập nhật (`walkthrough.md`).
