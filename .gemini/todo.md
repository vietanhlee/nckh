# Danh sách công việc (TODO List)

- [x] Khắc phục lỗi **CUDA Out of Memory (OOM)** trong `benchmark_5seeds.py`:
  - Thêm cấu hình PyTorch Allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Tự động xóa giải phóng các Tensor rác (`del X, Y, pred, y_true, y_pred, err, abs_err`) ngay trong vòng lặp batch.
  - Xóa mô hình, optimizer và thu gom rác bộ nhớ (`del model, optimizer; torch.cuda.empty_cache(); gc.collect()`) sau mỗi seed/mô hình.
  - Đổi `batch_size` mặc định về `32` (giảm tải memory GPU gấp 2 lần).
- [x] Tích hợp WandB (Weights & Biases) tự động cho `benchmark_5seeds.py`.
- [x] Cấu hình đường dẫn thư mục gốc động `ROOT_DIR`.
- [x] Thêm chức năng đo đạc MAE đầy đủ cho **CẢ 6 BƯỚC THỜI GIAN ($t+1 \rightarrow t+6$)**.
- [x] Cập nhật bảng kết quả trong `benchmark_5seeds.py` và báo cáo `benchmark_5seeds_report.md`.
- [x] Xuất báo cáo tài liệu kiến trúc ra `architecture_summary.md`.
- [x] Tích hợp lại mô hình `STGCN_BlockAttn` vào `benchmark_5seeds.py` và tối ưu hóa bộ nhớ CUDA (Chunking Multi-Head Attention & 2 Blocks) chống lỗi OOM trên GPU Nvidia T4.
- [x] Tích hợp mô hình `GCN_LSTM` (`ImprovedGNN_LSTM`) vào `benchmark_5seeds.py` và xếp vị trí chạy đầu tiên trong danh sách 5 mô hình thử nghiệm.

