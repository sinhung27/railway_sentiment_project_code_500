# Phân tích phản hồi hành khách đường sắt — bộ khung đồ án

## 1. Chức năng
- Huấn luyện và so sánh Multinomial Naive Bayes với Linear SVM.
- Biểu diễn văn bản bằng TF-IDF.
- Giao diện Streamlit để phân loại một phản hồi hoặc phân tích CSV hàng loạt.
- Xuất bảng kết quả, chỉ số đánh giá, ma trận nhầm lẫn và các câu mô hình dự đoán sai.

## 2. Cấu trúc
- `data/raw/feedback_raw.csv`: bộ 500 dòng dữ liệu mẫu tổng hợp được cung cấp kèm dự án.
- `data/sample_demo.csv`: dữ liệu minh họa nhỏ để kiểm tra chương trình.
- `train.py`: huấn luyện và đánh giá mô hình.
- `app.py`: giao diện demo.
- `models/`: nơi lưu mô hình và kết quả huấn luyện.
- `report_outline.md`: dàn ý báo cáo và nội dung cần ghi nhận.

## 3. Cài đặt trên Windows
Cài Python 3.11 hoặc 3.12 từ trang chính thức python.org. Trong lúc cài, chọn **Add Python to PATH**.

Mở thư mục dự án, nhấp thanh địa chỉ File Explorer, gõ `cmd` rồi nhấn Enter. Chạy lần lượt:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu lệnh `python` không hoạt động, thử `py` thay cho `python`.

## 4. Kiểm tra chương trình bằng dữ liệu minh họa
Dữ liệu trong `sample_demo.csv` được viết thủ công để kiểm tra luồng chạy. Không dùng điểm số từ dữ liệu này làm kết quả khoa học trong báo cáo.

```bat
python train.py
streamlit run app.py
```

Trình duyệt thường tự mở. Nếu không, mở địa chỉ hiển thị trong cửa sổ lệnh, thường là `http://localhost:8501`.

## 5. Huấn luyện với dữ liệu chính thức
Tạo `data/feedback.csv` với hai cột:
- `text`: nội dung phản hồi.
- `label`: một trong ba nhãn `tich_cuc`, `tieu_cuc`, `trung_tinh`.

Ví dụ:

```csv
text,label
"Nhân viên hướng dẫn tận tình",tich_cuc
"Tàu chậm, không có thông báo",tieu_cuc
"Chuyến tàu khởi hành lúc 8 giờ",trung_tinh
```

Sau đó chạy:

```bat
python train.py
streamlit run app.py
```

## 6. Dữ liệu và tính trung thực học thuật
UIT-VSFC là bộ dữ liệu phản hồi của sinh viên, không phải phản hồi của hành khách đường sắt. Có thể dùng nó để học quy trình phân loại cảm xúc tiếng Việt, nhưng không nên mô tả nó như dữ liệu đường sắt. Nếu dùng dữ liệu từ miền khác, cần ghi rõ đây là dữ liệu thay thế và nêu giới hạn khi áp dụng cho đường sắt.

Đối với đề tài này, nên ưu tiên phản hồi đường sắt công khai, ghi lại nguồn và ngày thu thập, tuân thủ điều khoản sử dụng, loại thông tin nhận dạng cá nhân, rồi gán nhãn theo quy tắc nhất quán. Nếu không tìm đủ dữ liệu, báo cáo rõ số lượng thực tế; không tự nhận dữ liệu minh họa là dữ liệu khảo sát thật.

## 7. Quy ước gán nhãn
- `tich_cuc`: khen ngợi hoặc thể hiện hài lòng rõ ràng.
- `tieu_cuc`: phàn nàn hoặc thể hiện không hài lòng rõ ràng.
- `trung_tinh`: thông báo sự kiện, thông tin trung lập hoặc không thể xác định cảm xúc rõ.
- Câu vừa khen vừa chê, mỉa mai, hoặc thiếu ngữ cảnh nên được xem xét thủ công và ghi quy tắc xử lý trước khi gán nhãn.

## 8. Lưu ý
- Kết quả thay đổi theo dữ liệu và cách chia tập; không cam kết trước độ chính xác.
- Dùng cùng một cách chia train/test khi so sánh mô hình.
- Không chọn mô hình chỉ dựa vào accuracy nếu các lớp mất cân bằng; xem cả macro-F1 và kết quả từng lớp.
- Chỉ mở `app.py` sau khi đã tạo `models/best_model.joblib` bằng `train.py`.


## 9. Bộ dữ liệu đi kèm dự án
File `data/raw/feedback_raw.csv` có 500 dòng, gồm 3 cột `text`, `label`, `topic`.
Đây là dữ liệu tổng hợp bằng AI phục vụ thử nghiệm kỹ thuật, không phải 500 phản hồi thực tế được thu thập từ hành khách. Nếu đề bài yêu cầu dữ liệu thực, hãy thay bằng dữ liệu có nguồn rõ ràng và cập nhật mô tả trong báo cáo.
