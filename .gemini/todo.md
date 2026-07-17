# Danh sách công việc (TODO List)

- [x] Sửa warning deprecation (`GradScaler`/`autocast`) và tích hợp `tqdm` (chuyển sang `tqdm.auto`) cho:
  - [x] [gcn_lstm.py](file:///g:/nckh/gcn_lstm.py)
  - [x] [wavenet_gcn.py](file:///g:/nckh/wavenet_gcn.py)
  - [x] [astgcn.py](file:///g:/nckh/astgcn.py)
- [x] Nâng early stopping (`PATIENCE`) lên 20 ở tất cả mô hình.
- [x] Xây dựng thêm các mô hình mới bằng PyTorch thuần:
  - [x] [stgcn.py](file:///g:/nckh/stgcn.py) [NEW]
  - [x] [dcrnn.py](file:///g:/nckh/dcrnn.py) [NEW]
  - [x] [stgcn_gcn.py](file:///g:/nckh/stgcn_gcn.py) [NEW]
- [x] Chuyển đổi toàn bộ thư mục checkpoint `SAVE_DIR` sang thư mục tương đối cục bộ `"model/"` ở tất cả mô hình.
- [x] Cập nhật script [compare_models.py](file:///g:/nckh/compare_models.py) để tích hợp cả 6 mô hình và tự động tạo thư mục checkpoint `model/`.
- [x] Cập nhật tài liệu [README.md](file:///g:/nckh/README.md).
- [x] Kiểm tra và chạy thực nghiệm xác minh.
