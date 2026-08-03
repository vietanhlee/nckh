# 📝 Đề xuất Cải thiện & Mở rộng Nội dung Bài báo Khoa học (Paper Enhancement Recommendations)

Tài liệu này tổng hợp các hạng mục gợi ý cải thiện, bổ sung thực nghiệm và nâng cấp bài báo khoa học nhằm tối ưu hóa khả năng được chấp nhận (acceptance rate) tại các tạp chí/hội nghị top-tier về Hệ thống Giao thông Thông minh (như *IEEE Transactions on Intelligent Transportation Systems*, *IEEE TKDE*, *AAAI*).

---

## 🎯 1. Các Hạng mục đã Cải thiện Hoàn chỉnh (Completed Enhancements)

- [x] **Cấu trúc 2 Bài toán con (Two-Subproblem Framework)**: Đã bóc tách rõ ràng Sub-problem 1 (Camera-level Perception - ConvNeXt-Tiny) và Sub-problem 2 (Network-wide Graph Forecast - TA-STGCN).
- [x] **Lập luận Toán học & Lý do Mô hình hóa Đồ thị**: Đã phân tích lý do dự báo độc lập từng nút (Single-node forecasting) bị thất bại do hiện tượng lan truyền ùn tắc (Traffic Spillover), bắt buộc phải dùng Đồ thị 608 nodes $\mathcal{G} = (\mathcal{V}, \mathcal{E}, W)$.
- [x] **Số liệu 5 Seeds Thực nghiệm Chuẩn xác**: Đã cập nhật kết quả 5 seeds ngẫu nhiên ($\text{Mean} \pm \text{Std}$) cho 3 mô hình (`GCN-LSTM`, `STGCN Baseline`, `TA-STGCN`).
- [x] **Định dạng Bảng biểu (Tables Format)**: Đã dùng `table*` hai cột và `resizebox` giải quyết dứt điểm lỗi tràn lề (overflow) của Bảng 1 và Bảng 3.
- [x] **So sánh Chi tiết Kiến trúc Vision (Stage 1)**: Đã bổ sung so sánh chi tiết giữa ResNet-50 (Residual Bottleneck, BN, 3x3 Conv) và ConvNeXt-Tiny (Patchify Stem, 7x7 Depthwise Conv, Inverted Bottleneck, LayerNorm, GELU).
- [x] **Tích hợp Đầy đủ 7 Hình ảnh Thực tế trong `paper/fig/`**:
  - `traffic_jam.jpg` (Hình 1 - Bối cảnh ùn tắc giao thông TP.HCM).
  - `samples_pictures_get_from_api.png` (Hình 2 - Ảnh camera API HCM Traffic Portal).
  - `stgcn_architecture.png` (Hình 3 - Sơ đồ kiến trúc TA-STGCN).
  - `results_training_model_counting.png` (Hình 4 - Đường cong hội tụ ConvNeXt-Tiny).
  - `attention_map.png` (Hình 5 - Visual attention heatmaps đếm xe).
  - `fig_compare_mae_curve_models_graph.png` (Hình 6 - Đường Val MAE so sánh tốc độ hội tụ & chống nhiễu).
  - `result_of_a_node_by_ours_model.png` (Hình 7 - Đồ thị dự báo thực tế vs dự báo trên 1 node).

---

## 🚀 2. Đề xuất Mở rộng & Bổ sung Cho Các Phiên bản Tiếp theo (Future Paper Extensions)

### A. Phân tích Ablation Study (Thực nghiệm Loại trừ)
Để làm rõ hơn đóng góp của từng thành phần trong `TA-STGCN`, có thể bổ sung 1 bảng Ablation Study nhỏ trong Section V:
1. `TA-STGCN (Full Model)`: Đầy đủ 2 khối ST-Conv + Model-Level Multi-Head Attention.
2. `TA-STGCN w/o Attention`: Gỡ bỏ khối Attention ở cuối (chỉ còn 2 khối ST-Conv).
3. `TA-STGCN w/ Single-Head Attn`: Thay Multi-Head Attention ($h=4$) bằng Single-Head Attention ($h=1$).
4. `TA-STGCN with Dynamic Adj`: Thay ma trận kề tĩnh RBF $W$ bằng ma trận kề tự học động (Adaptive Adjacency Matrix).

### B. Mở rộng Tập Dữ liệu So sánh Quốc tế (Public Benchmarks)
- Bài báo hiện tại tập trung thực nghiệm trên tập dữ liệu thực tế tại TP.HCM (608 nodes). Để tăng tính tổng quát quốc tế, có thể thử nghiệm thêm `TA-STGCN` trên 2 tập dữ liệu công khai tiêu chuẩn: **PeMSD4** và **PeMSD8** (California Highway Traffic).

### C. Đánh giá Mức độ Tiêu thụ Tài nguyên Tính toán (Edge Computing Efficiency)
- Đưa thêm chỉ số về FLOPs, Memory Allocation (MB/batch) và Tốc độ suy luận thực tế (FPS / Latency ms) của cả Stage 1 (ConvNeXt-Tiny) và Stage 2 (TA-STGCN) trên các thiết bị tính toán nhúng (như NVIDIA Jetson Orin / Xavier) để chứng minh tính khả thi khi triển khai Edge AI tại camera giao thông.

---

## 📌 3. Tóm tắt Trạng thái Bài báo
- **Tệp LaTeX chính**: [paper/main.tex](file:///g:/nckh/paper/main.tex)
- **Hình ảnh**: Đã cập nhật 7 tệp hình ảnh vào `paper/fig/`.
- **Trạng thái**: Bài báo đạt chuẩn trình bày khoa học quốc tế IEEEtran, mạch lạc, đầy đủ công thức, không tràn lề và sẵn sàng xuất bản.
