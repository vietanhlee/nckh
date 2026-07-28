# Kế hoạch Thực hiện: Công cụ Relabel Data Giao thông (traffic_final_windows_order.csv)

Xây dựng một công cụ GUI bằng Python Tkinter (`relabel_tool.py`) hỗ trợ xem ảnh, đọc nhãn hiện tại, lọc dữ liệu theo nhãn (1-5) và gán lại nhãn nhanh bằng phím tắt / nút bấm.

---

## 🛠️ Thiết kế Công cụ `relabel_tool.py`

### 1. Đầu vào & Cấu hình mặc định
Dựa trên lines 738-739 của `coral_focal.py`:
- `csv_file`: `/kaggle/input/datasets/huecute/csv-images/traffic_final_windows_order.csv`
- `image_dir`: `/kaggle/input/datasets/huecute/images/images`
- Có giao diện điều chỉnh đường dẫn (Browse File CSV, Browse Directory Ảnh) linh hoạt khi chạy cục bộ.

### 2. Các chức năng chính
- **Bộ lọc nhãn (Label Filter)**:
  - Chọn xem "Tất cả" hoặc lọc chỉ xem các mẫu thuộc Nhãn 1, Nhãn 2, Nhãn 3, Nhãn 4, Nhãn 5.
- **Xem ảnh & Thông tin chi tiết**:
  - Tự động thay đổi kích thước ảnh tỉ lệ chuẩn theo khung giao diện.
  - Hiển thị tên file (`filename`), nhãn gốc, nhãn hiện tại (`phan_loai`), chỉ số dòng trong CSV.
  - Thống kê tổng quan số lượng mẫu theo từng nhãn realtime.
- **Đánh nhãn nhanh (Quick Relabel)**:
  - Nút bấm trực quan cho 5 mức nhãn (1 -> 5).
  - **Phím tắt**: Nhấn `1`, `2`, `3`, `4`, `5` để gán nhãn tức thì và tự động chuyển sang ảnh kế tiếp.
- **Điều hướng linh hoạt**:
  - Nút `Previous`, `Next` hoặc phím mũi tên `Left`, `Right` / `A`, `D`.
  - Ô nhập số thứ tự để nhảy nhanh tới một ảnh bất kỳ.
- **Lưu dữ liệu**:
  - Ghi đè file CSV (tạo file backup `.bak`).
  - Cho phép xuất file CSV mới (`Save As...`).

---

## 📅 Các tệp sẽ thay đổi/thêm mới

* #### [NEW] [relabel_tool.py](file:///g:/nckh/relabel_tool.py)
  Script chính chứa giao diện Tkinter + Pandas + PIL.
* #### [MODIFY] [.gemini/todo.md](file:///g:/nckh/.gemini/todo.md)
  Bổ sung mục quản lý công việc gán lại nhãn dữ liệu.

---

## 🔍 Kế hoạch Xác minh

### Kiểm tra tự động
- Kiểm tra cú pháp script `relabel_tool.py`.

### Kiểm tra thủ công (Thực hiện bởi người dùng)
- Chạy lệnh: `python relabel_tool.py`
- Kiểm tra tính năng chọn file, lọc nhãn, phím tắt 1-5, lưu file CSV.
