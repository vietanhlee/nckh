# Danh sách công việc (TODO List)

- [x] Sửa warning deprecation (`GradScaler`/`autocast`) và tích hợp `tqdm` (chuyển sang `tqdm.auto`) cho:
  - [x] [gcn_lstm.py](file:///g:/nckh/gcn_lstm.py)
  - [x] [wavenet_gcn.py](file:///g:/nckh/wavenet_gcn.py)
- [x] Nâng early stopping (`PATIENCE`) lên 20 ở tất cả mô hình.
- [x] Xây dựng các mô hình mới dạng cải tiến của DCRNN:
  - [x] [dcrnn_glu.py](file:///g:/nckh/dcrnn_glu.py) [NEW]
  - [x] [dcrnn_bilstm.py](file:///g:/nckh/dcrnn_bilstm.py) [NEW]
  - [x] [dcrnn_tcn.py](file:///g:/nckh/dcrnn_tcn.py) [NEW]
  - [x] [dcrnn_attention.py](file:///g:/nckh/dcrnn_attention.py) [NEW]
- [x] Chuyển đổi toàn bộ thư mục checkpoint `SAVE_DIR` sang thư mục tương đối cục bộ `"model/"` ở tất cả mô hình.
- [x] Cập nhật script [compare_models.py](file:///g:/nckh/compare_models.py) để tích hợp cả 7 mô hình và tự động tạo thư mục checkpoint `model/`.
- [x] Cập nhật tài liệu [README.md](file:///g:/nckh/README.md).
- [x] Kiểm tra và chạy thực nghiệm xác minh.
