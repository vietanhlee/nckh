# Danh sách công việc (TODO List)

- [x] Tích hợp WandB (Weights & Biases) cho các mô hình.
- [x] Cấu hình đường dẫn thư mục gốc động `ROOT_DIR`.
- [x] **[MỚI]** Thêm chức năng đo đạc MAE đầy đủ cho **CẢ 6 BƯỚC THỜI GIAN ($t+1 \rightarrow t+6$)**:
  - `MAE t+1`: Dự báo sau 5 phút
  - `MAE t+2`: Dự báo sau 10 phút
  - `MAE t+3`: Dự báo sau 15 phút
  - `MAE t+4`: Dự báo sau 20 phút
  - `MAE t+5`: Dự báo sau 25 phút
  - `MAE t+6`: Dự báo sau 30 phút
- [x] Cập nhật bảng kết quả trong `benchmark_5seeds.py` và báo cáo `benchmark_5seeds_report.md` xuất đủ 6 cột horizon.
- [x] Cập nhật hàm `evaluate` và in kết quả chi tiết trong tất cả các script mô hình lẻ (`stgcn.py`, `hybrid.py`, `stgcn_block_attn.py`, `stgcn_mixed_blocks.py`).
- [x] Xuất báo cáo tài liệu kiến trúc ra `architecture_summary.md`.
