# Draft Notes

Nguồn LaTeX chuẩn của bài báo là [latex_paper.tex](latex_paper.tex). Tài liệu này giữ vai trò tóm tắt định hướng để tránh tồn tại hai bản thảo có nội dung mâu thuẫn.

## Paper Framing

Bài báo không xem nhận diện gỗ chỉ là bài toán chọn backbone hoặc hàm loss. Vấn đề trung tâm là một mẫu vật vật lý thường sinh nhiều ảnh tương quan. Do đó, nếu các ảnh được chia độc lập, đánh giá có thể đo khả năng nhận lại dấu vết của mẫu vật hay phiên chụp thay vì nhận diện một mẫu vật mới.

Đóng góp chính là Specimen-Centric Data Protocol (SCDP), bao phủ:

- xác định đơn vị nguồn cần tổng quát hóa;
- thu thập và lưu metadata truy vết;
- tổ chức ảnh gốc, ảnh dẫn xuất và manifest;
- kiểm toán nhãn, trùng lặp và tính toàn vẹn nhóm;
- phân bổ group-disjoint, phát triển mô hình trên train--validation, và khóa test trước khi đánh giá.

CEGS-Split là thành phần phân bổ của SCDP. Nó bảo toàn toàn bộ ảnh của một mẫu vật trong cùng một tập, cân bằng theo loài và dùng embedding đóng băng để gắn cờ các cặp liên tập cần kiểm tra. Embedding similarity là tín hiệu kiểm toán, không phải bằng chứng tự thân của rò rỉ.

## Result Interpretation

Kết quả đối chiếu giữa random image split và CEGS-Split được trình bày như một phân tích độ nhạy theo đơn vị phân chia. Chênh lệch điểm số không được diễn giải như một hằng số về mức độ đánh giá quá cao cho mọi bộ dữ liệu. Một kết luận về rò rỉ phải dựa trên giao giữa định danh nguồn của các tập và manifest có thể kiểm tra.

## Submission Checklist

Trước khi nộp, cần phát hành hoặc mô tả đầy đủ manifest S3, seed và tỷ lệ split; chạy nhiều split group-disjoint hoặc group cross-validation; và bổ sung external test set nếu mục tiêu là khẳng định khả năng tổng quát hóa ngoài nguồn thu thập.
