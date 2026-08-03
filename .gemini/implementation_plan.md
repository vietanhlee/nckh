# Kế hoạch Thực hiện: Cấu trúc lại Bài báo Khoa học theo 2 Bài toán con (Sub-Problems)

Cấu trúc lại toàn bộ bài báo khoa học LaTeX (IEEEtran) trong `g:/nckh/paper/` theo đúng luồng tư duy nghiên cứu 2 bài toán con phân tách rõ ràng.

---

## 🎯 Luồng Nội dung Nghiên cứu
1. **Bối cảnh Tổng quan**: Quản lý giao thông đô thị thông minh (ITS), bùng nổ phương tiện cá nhân, ưu thế của mạng lưới camera CCTV sẵn có so với cảm biến vật lý đắt đỏ.
2. **Phân tách thành 2 Bài toán con (Sub-Problems)**:
   - **Bài toán con 1 (Camera-Level Vehicle Count Estimation)**: Ước lượng lưu lượng phương tiện phân loại [Ô tô, Xe máy] tại từng camera/tuyến đường bằng phương pháp Direct Regression với **ConvNeXt-Tiny** (MAE ≈ 5.0 trên 5,018 ảnh từ 657 camera TP.HCM).
   - **Bài toán con 2 (Network-Wide Traffic Flow Forecasting)**: Lập luận khoa học vì sao dự báo độc lập từng nút (Single-node forecasting) là **bất khả thi và không hiệu quả** do hiện tượng lan truyền ùn tắc (Traffic Spillover). Cần mô hình hóa toàn bộ Đồ thị Không-Thời gian $\mathcal{G} = (\mathcal{V}, \mathcal{E}, W)$.
3. **3 Mô hình đối chứng cho Bài toán con 2**:
   - **Baseline 1**: GCN + LSTM (`gcn_lstm.py`) — GCN tĩnh 2 tầng + LSTM 2 tầng.
   - **Baseline 2**: STGCN gốc (`stgcn.py`) — Chebyshev Graph Conv ($K=3$) + 1D Temporal GLU.
   - **Proposed (Our Model)**: STGCN-Hybrid (`hybrid.py`) — Chebyshev Graph Conv ($K=3$) + 1D GLU + Model-Level Multi-Head Temporal Self-Attention.

---

## 🛠️ Các tệp sẽ cập nhật
- `paper/latex_paper.tex`
- `paper/sections/introduction.tex`
- `paper/sections/methodology.tex`
- `paper/sections/baselines.tex`
- `paper/sections/experiments.tex`
- `paper/sections/results.tex`
- `paper/sections/explainability.tex`
- `paper/sections/conclusion.tex`
- `paper/sections/references.tex`
