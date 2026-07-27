# Danh sách công việc (TODO List)

- [x] Tích hợp WandB (Weights & Biases) để log metric cho từng mô hình khi chạy riêng lẻ.
- [x] Tích hợp WandB vào `compare_models.py`.
- [x] Cấu hình đường dẫn thư mục gốc động `ROOT_DIR` cho tất cả các mô hình.
- [x] **[MỚI]** Cải tiến kiến trúc `hybrid.py`:
  - Thay GRU "làm mượt thụ động" bằng **Multi-Head Temporal Self-Attention** (`TemporalAttention` + residual connection + FFN).
  - Tăng capacity của STGCN backbone: `BLOCK_HIDDEN=64`, `NUM_BLOCKS=3`, `CHEB_K=3`, `DROPOUT=0.25`.
  - Tinh chỉnh hyperparameters training: `BATCH_SIZE=64`, `LEARNING_RATE=0.0005`, `PATIENCE=60`, giữ nguyên `LOSS_DELTA=1.0`.
- [x] Tạo tài liệu hướng dẫn và báo cáo cập nhật (`walkthrough.md`).
