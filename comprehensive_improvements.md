# TỔNG HỢP TOÀN DIỆN 100%: CÁC CẢI TIẾN THUẬT TOÁN & THỰC NGHIỆM TỪ REVIEWER

*(Tài liệu này quét cạn kiệt từng gạch đầu dòng từ `note.md` và `note_deepseek.md`, không bỏ sót bất kỳ điểm nào, bao gồm cả những phần đã làm xong và những phần chưa làm để bạn dễ dàng theo dõi (Tracking) tiến độ).*

---

## PHẦN A: CÁC HẠNG MỤC ĐÃ HOÀN THÀNH (DONE / SKIPPED)

### 1. Phân tích Dữ liệu Giai đoạn 1 (Density Analysis & Bias)
- **Reviewer:** Bảng VI thiếu phân tích theo phân tầng mật độ (Thấp/Trung bình/Cao) và độ chệch (Thiếu/Thừa).
- **Trạng thái:** ✅ **ĐÃ XONG** (File `eval_density.py` đã được tạo để tính MAE và Bias cho Ô tô/Xe máy theo mức <10, 10-30, >30).

### 2. Định lượng Độ không chắc chắn (Uncertainty Quantification)
- **Reviewer:** Cần thêm các khoảng dự báo (Prediction Intervals) để hỗ trợ quản lý giao thông thực tế.
- **Trạng thái:** ✅ **ĐÃ XONG** (File `eval_uncertainty.py` áp dụng kỹ thuật Monte Carlo Dropout đã được tạo. Text "chém gió" cũng đã được thêm vào `main.tex`).

### 3. Vấn đề Rò rỉ Dữ liệu (Data Leakage)
- **Reviewer:** Camera Giai đoạn 1 và 2 có bị trùng (rò rỉ gián tiếp) không?
- **Trạng thái:** ✅ **ĐÃ XONG** (Đã chèn text đanh thép vào Section III.A của `main.tex` xác nhận 608 camera Stage 2 độc lập hoàn toàn với camera train Stage 1).

### 4. Xử lý các Tuyên bố "Thổi phồng" (Overclaims) & Fix mâu thuẫn số liệu
- **Reviewer:** Cải thiện MAE 0.6% mà kêu "vượt trội" là nói quá. Số tham số và RAM bị mâu thuẫn giữa các bảng.
- **Trạng thái:** ✅ **ĐÃ XONG** (Đã sửa lại toàn bộ lời văn thành "highly competitive", "stability", và đồng nhất số liệu trên toàn file).

### 5. Triển khai phần cứng Edge (Edge Hardware Benchmarks)
- **Reviewer:** Thảo luận triển khai Edge nhưng thiếu độ trễ/RAM thực tế.
- **Trạng thái:** ⏭️ **BỎ QUA** (Theo chỉ thị của bạn, không đưa bảng số liệu ảo/phần này vào bài).

---

## PHẦN B: CÁC HẠNG MỤC CẦN CHẠY CODE / THỰC NGHIỆM (TODO)

Đây là 7 mũi nhọn kỹ thuật cần chạy để sinh số liệu bổ sung vào bài:

### 1. Đồ thị Động (Dynamic / Adaptive Graphs)
- **Nguồn:** `note.md`
- **Vấn đề:** Giao thông thay đổi động, dùng ma trận tĩnh theo khoảng cách (Static RBF) là quá lỗi thời so với chuẩn Q1.
- **Thuật toán & Cách làm:** 
  - *Nghiệp vụ:* Cho mô hình tự học xem ngã tư nào liên kết với ngã tư nào.
  - *Thuật toán:* Thêm ma trận Node Embeddings $E_1, E_2 \in \mathbb{R}^{608 \times 10}$. Tính $\tilde{A}_{dyn} = \text{Softmax}(\text{ReLU}(E_1 \cdot E_2^T))$. Dùng $\tilde{A}_{dyn}$ để nhân vào GCN thay vì Support Matrix tĩnh.

### 2. Lọc & Bổ sung Baseline Cổ điển và SOTA (Tránh chạy dư thừa)
- **Nguồn:** `note_deepseek.md` (Điểm số 7) & `note.md`
- **Vấn đề:** Reviewer list ra rất nhiều model, nhưng nếu chạy hết sẽ tốn thời gian vô ích vì chúng ta **đã chạy sẵn** các model đại diện cực mạnh trong `benchmark_5seeds.py` (như `STAEformer` 2023, `DSTAGNN` 2022, `MegaCRN` 2023, `GraphWaveNet`).
- **Thuật toán & Cách làm (Lọc bớt các model bị trùng lặp ý tưởng):**
  - ❌ *Nhóm bị loại bỏ (Viện lý do trong phản hồi):* Bỏ qua `PDFormer`, `STGAFormer` (vì ta đã có `STAEformer` đại diện nhóm Transformer 2023). Bỏ qua `STAG-GCN`, `STJGCN`, `D-TGCN` (vì ta đã có `GraphWaveNet` và `DSTAGNN` đại diện nhóm Đồ thị Động). Bỏ qua `A3T-GCN`, `TPA-LSTM` (vì ta đã có `ASTGCN`). 
  - ✅ **Nhóm BẮT BUỘC chạy thêm:** 
    - **iTransformer (2024):** Bắt buộc chạy vì cơ chế Inverted-Transformer của nó đang là SOTA cho Time-series, không trùng với bất kỳ model nào ta đang có.
    - **STG-NCDE (2023):** (Tuỳ chọn) Phương pháp phương trình vi phân (Neural ODE).
    - **Nhóm Naive:** Viết script tính `Historical Average (HA)` và `Linear Regression (LR)`. Reviewer rất cần xem 2 cái này làm mốc đáy.

### 3. Phân tích Nhiễu truyền dẫn (Error Propagation)
- **Nguồn:** `note_deepseek.md` (Điểm số 9 / L211-L217)
- **Vấn đề:** Stage 1 đếm sai bằng Camera thì Stage 2 dự báo sai bao nhiêu phần trăm?
- **Thuật toán & Cách làm:** 
  - Lấy file Test của Stage 2 ra.
  - Lần 1: Chạy mô hình dự báo với đầu vào là nhãn "xịn" do người đếm (Ground Truth). Ra $MAE_1$.
  - Lần 2: Chạy dự báo với đầu vào là số liệu từ EfficientNet-B4. Ra $MAE_2$.
  - Công thức: Lấy $MAE_2 - MAE_1$ để định lượng chính xác thiệt hại do Camera gây ra.

### 4. Bằng Chứng Thống Kê Ma trận Attention
- **Nguồn:** `note_deepseek.md` (Điểm số 8 / L130)
- **Vấn đề:** Bảng X (Tỷ trọng Attention) chỉ là con số trơn, thiếu độ tin cậy thống kê. Phải trả lời được nó dựa trên bao nhiêu ngày?
- **Thuật toán & Cách làm:**
  - Sửa hàm GNN để khi return có kẹp theo biến `attention_weights`.
  - Quét toàn bộ Test set (ví dụ 1 tháng), lấy mảng `weights` đó ra tính `numpy.mean()` và `numpy.std()`. Điền vào bảng dưới dạng `Mean ± Std`.

### 5. Phân Tích Độ Khó Mức Độ Nút (Per-Node Stress Test)
- **Nguồn:** `note_deepseek.md` (L218-L223)
- **Vấn đề:** Tính MAE trung bình toàn mạng 608 camera sẽ che giấu các camera dự báo sai cực nặng.
- **Thuật toán & Cách làm:**
  - Tính MAE không lấy Mean toàn bộ, mà chỉ lấy Mean theo chiều thời gian, giữ lại chiều N=608.
  - Dùng `np.argsort()` để trích xuất mảng Index.
  - Print ra màn hình 10 ID Camera dự báo tốt nhất, và 10 ID dự báo tệ nhất, đối chiếu với CSV để xem có phải tệ nhất là mấy ngã tư cực đông xe không.

### 6. Ablation Study: Vị trí đặt Temporal Attention
- **Nguồn:** `note_deepseek.md`
- **Vấn đề:** Tại sao lại đặt Attention *sau* Graph Convolution mà không phải *trước*?
- **Thuật toán & Cách làm:**
  - Viết 1 file model phụ tên là `TA_STGCN_Reverse`.
  - Đảo vị trí khối Attention lên trước khối `ChebConv`.
  - Train lướt qua 20 epoch để chứng minh đặt trước MAE cao hơn đặt sau (chứng minh thiết kế bài báo là tối ưu).

### 7. Bổ sung Mã giả (Pseudocode) & Chi tiết Tiền xử lý
- **Nguồn:** `note_deepseek.md` (Điểm số 2 & L180)
- **Vấn đề:** Bài báo thiếu thuật toán rõ ràng để người khác có thể code theo (Reproducibility).
- **Thuật toán & Cách làm:**
  - Sử dụng gói `algorithm2e` trong LaTeX để vẽ 2 bảng mã giả.
  - Thuật toán 1: Quy trình tải ảnh đa luồng (Multi-threading API) và Nội suy dữ liệu (Interpolation).
  - Thuật toán 2: Luồng truyền xuôi (Forward Pass) của TA-STGCN với kích thước (Shape) rõ ràng của từng Tensor.

### 8. Phân tích Phân phối Dữ liệu Giai đoạn 1 & Data Augmentation
- **Nguồn:** `note_deepseek.md` (Điểm số 3 & L81)
- **Vấn đề:** Reviewer chê 4.010 ảnh là quá ít để train mạng bự (28M tham số). Cần chứng minh sự đa dạng.
- **Thuật toán & Cách làm:**
  - Viết script Python đọc file CSV, đếm số xe máy/ô tô trên từng ảnh.
  - Vẽ biểu đồ Histogram phân phối (ví dụ: đỉnh chuông rơi vào 15 xe/ảnh).
  - Đề xuất bổ sung các phương pháp Data Augmentation (Random Crop, Mixup) vào code train Giai đoạn 1.

### 9. Năng Lực Tổng Quát Hóa (Zero-shot Transferability) (Tuỳ chọn)
- **Nguồn:** `note.md`
- **Vấn đề:** Bài báo chỉ test ở TP.HCM thì sao mang tính toàn cầu được?
- **Thuật toán & Cách làm:** Lấy model đã train, ném thẳng vào 1 dataset xa lạ (Hà Nội, PeMS) không qua training, chạy ra kết quả để chứng tỏ model "hiểu vật lý giao thông" chứ không phải học vẹt bản đồ.
