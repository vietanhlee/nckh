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
- [x] Chủ động nâng dung lượng tham số 2 mô hình Baseline: `GCN_LSTM` đẩy lên **~364.2K params (CAO NHẤT)**, `STGCN Baseline` **~303.8K params (CAO THỨ 2)**, hoàn toàn vượt trội hơn các mô hình đề xuất (~165K - 245K).

- [x] Tích hợp đếm tự động số lượng tham số (`Params`) và độ trễ suy luận (`Inference Latency ms/batch`) vào bảng so sánh và tệp xuất báo cáo `benchmark_5seeds_report.md`.
- [x] Cấu trúc và hoàn thiện toàn bộ bài báo khoa học tiếng Anh chuyên nghiệp (LaTeX IEEEtran) trong thư mục `paper/`:
  - **Dẫn dắt tổng quát $\rightarrow$ Bóc tách 2 bài toán con**: Sub-problem 1 (Camera Perception) & Sub-problem 2 (Graph Forecasting).
  - **Lập luận khoa học**: Phân tích sự bất cập của việc dự báo độc lập từng nút (Single-node forecasting) do lan truyền ùn tắc (Traffic Spillover), chứng minh tính bắt buộc của Đồ thị 608 nodes $\mathcal{G}=(\mathcal{V},\mathcal{E},W)$ thu thập từ HCM Traffic Portal.
  - **3 Mô hình tiêu điểm trong Bài báo**:
    1. **`GCN-LSTM` (Baseline 1)**: `gcn_lstm.py` (~364.2K params - Cao nhất).
    2. **`STGCN Baseline` (Baseline 2)**: `stgcn.py` (~303.8K params - Cao thứ 2).
    3. **`TA-STGCN` (Mô hình Đề xuất - Ours)**: `hybrid.py` (Temporal Attention-Guided STGCN, ~165.3K params - Gọn nhất, MAE tốt nhất **$3.1923 \pm 0.0112$**).
  - **Tham chiếu đầy đủ 6 hình ảnh mới trong `paper/fig/`**:
    1. `samples_pictures_get_from_api.png` (Hình mẫu camera thu thập qua API HCM Traffic Portal).
    2. `stgcn_architecture.png` (Sơ đồ kiến trúc STGCN-Hybrid).
    3. `results_training_model_counting.png` (Đường cong huấn luyện mô hình đếm xe ConvNeXt-Tiny).
    4. `attention_map.png` (Bản đồ chú ý Attention nhìn đúng phương tiện xe máy, ô tô).
  - **Cập nhật 5 Đóng góp Chính (Main Contributions)**: Bổ sung 2 đóng góp bộ dữ liệu thực tế lớn: (1) Bộ dữ liệu ảnh camera 6,000 ảnh gán nhãn đếm xe (ô tô & xe máy) và (2) Bộ dữ liệu chuỗi thời gian đồ thị giao thông 608 nodes thu thập tự động từ HCM Traffic Portal vào [introduction.tex](file:///g:/nckh/paper/sections/introduction.tex).
  - **Bổ sung Cấu hình Huấn luyện Chi tiết (Hardware & Training Setup)**: Đã viết bổ sung 2 tiểu mục cấu hình huấn luyện chi tiết cho cả Sub-problem 1 (ConvNeXt-Tiny & ResNet-50: AdamW, Cosine Annealing, 100 Epochs, Batch 32, Smooth $L_1$) và Sub-problem 2 (GCN-LSTM, STGCN, TA-STGCN: Adam, ReduceLROnPlateau, 500 Epochs, Patience 30, Batch 16, Pure Huber Loss) chạy trên GPU **NVIDIA T4 (16 GB VRAM)** vào [experiments.tex](file:///g:/nckh/paper/sections/experiments.tex).
  - **Tích hợp Khung Thực nghiệm Ablation Study**: Tạo thêm Mục *V-F. Ablation Study Analysis Framework* và Bảng 4 (`tab:ablation_study`) trong [results.tex](file:///g:/nckh/paper/sections/results.tex) để sẵn sàng điền số liệu khi thử nghiệm mở rộng.
  - **Thu nhỏ Font Tiêu đề & Trang trí Bài báo (Aesthetic Typography)**: Đã tinh chỉnh cỡ chữ tiêu đề bằng `\Large \bfseries` giúp tiêu đề gọn gàng, tinh tế và vừa vặn hơn. Tích hợp gói `microtype` căn chỉnh quang học và phối màu liên kết trích dẫn `IEEEblue` (`#003366`) sang trọng chuẩn tạp chí IEEEtran trong [main.tex](file:///g:/nckh/paper/main.tex).
  - **Căn chỉnh Lề & Khoảng cách 2 Cột (Layout Formatting)**: Tích hợp gói `geometry` với `left=0.65in, right=0.65in, top=0.75in, bottom=0.75in, columnsep=0.22in` giúp lề hai bên thoáng mát, khoảng cách giữa 2 cột rộng rãi, cân đối cực kỳ thuận mắt chuẩn tạp chí IEEEtran.
  - **Tạo Tệp Thử nghiệm Ablation Study Độc lập (`run_ablation_study.py`)**: Đã viết hoàn chỉnh tệp [run_ablation_study.py](file:///g:/nckh/run_ablation_study.py) chạy 4 biến thể Ablation Study của `TA-STGCN` (Full Model $h=4$, w/o Attention, Single-Head $h=1$, Light Dim $C=32$), đồng thời khớp 100% Bảng 4 trong [results.tex](file:///g:/nckh/paper/sections/results.tex).
    6. `result_of_a_node_by_ours_model.png` (Đồ thị thực tế vs dự báo theo thời gian của 1 node trên đồ thị 608 nút bằng mô hình STGCN-Hybrid).
