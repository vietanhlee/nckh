# Kế hoạch Thực hiện: So sánh thực nghiệm 7 kiến trúc Spatial-Temporal Graph (NCKH)

Dự án nghiên cứu khoa học tập trung so sánh 7 kiến trúc đồ thị không-thời gian bằng PyTorch thuần để phục vụ dự báo lưu lượng giao thông, sử dụng thư mục checkpoint cục bộ để tăng tính linh hoạt.

---

## 🛠️ Thiết kế Kiến trúc các Mô hình

### 1. Các baseline chính
* **GCN-LSTM** ([gcn_lstm.py](file:///g:/nckh/gcn_lstm.py))
  * Kết hợp GCN tĩnh và LSTM.
* **Graph WaveNet** ([wavenet_gcn.py](file:///g:/nckh/wavenet_gcn.py))
  * GCN tĩnh + ma trận thích ứng động cùng Gated TCN.
* **STGCN** ([stgcn.py](file:///g:/nckh/stgcn.py))
  * Tích chập phổ Chebyshev (ChebNet) kết hợp Gated CNN (GLU) 1D.

### 2. Các biến thể học thời gian của DCRNN (Diffusion Graph Conv)
* **DCRNN-GLU** ([dcrnn_glu.py](file:///g:/nckh/dcrnn_glu.py)) [NEW]
  * Diffusion Graph Conv kết hợp **Gated CNN (GLU) 1D** học thời gian song song.
* **DCRNN-BiLSTM** ([dcrnn_bilstm.py](file:///g:/nckh/dcrnn_bilstm.py)) [NEW]
  * Diffusion Graph Conv kết hợp **Bidirectional LSTM (BiLSTM)** học chuỗi thời gian hai chiều.
* **DCRNN-TCN** ([dcrnn_tcn.py](file:///g:/nckh/dcrnn_tcn.py)) [NEW]
  * Diffusion Graph Conv kết hợp **Stacked Dilated Causal TCN** học quan hệ thời gian dài hạn.
* **DCRNN-Attention** ([dcrnn_attention.py](file:///g:/nckh/dcrnn_attention.py)) [NEW]
  * Diffusion Graph Conv kết hợp **Multi-Head Temporal Self-Attention** trích xuất ngữ cảnh động toàn cục.

---

## 📅 Các tệp đã thay đổi cụ thể

### [New Models & Checkpoint path updates]
* #### [NEW] [dcrnn_bilstm.py](file:///g:/nckh/dcrnn_bilstm.py)
* #### [NEW] [dcrnn_tcn.py](file:///g:/nckh/dcrnn_tcn.py)
* #### [NEW] [dcrnn_attention.py](file:///g:/nckh/dcrnn_attention.py)
* #### [MODIFY] Tất cả 7 tệp mô hình
  Chuyển đổi tham số đường dẫn lưu mô hình `SAVE_DIR` sang thư mục tương đối `"model/"`.

### [Comparison & Documentation]
* #### [MODIFY] [compare_models.py](file:///g:/nckh/compare_models.py)
  Tích hợp các mô hình thành **7 mô hình** so sánh tổng thể và tự động tạo thư mục checkpoint `model/` nếu chưa có.
* #### [MODIFY] [README.md](file:///g:/nckh/README.md)
  Bổ sung mô tả và cách chạy của 7 mô hình.

---

## 🔍 Kế hoạch Xác minh (Verification Plan)

### Kiểm tra tự động
- Chạy thử `python compare_models.py --mode train --epochs 1` để kiểm tra khả năng chạy tuần tự và tổng hợp kết quả của cả **7 mô hình**.

### Kiểm tra thủ công
- Xác nhận thư mục `./model/` được tự sinh tự động tại thư mục dự án và lưu trữ các file checkpoint `.pth` tốt nhất.
- Xác nhận bảng so sánh kết quả in ra terminal hiển thị đầy đủ thông tin của 7 mô hình.
- Xác nhận báo cáo `comparison_report.md` được ghi nhận chính xác.
