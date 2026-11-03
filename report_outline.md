# Dàn ý báo cáo đồ án

## Tên đề tài
Phân tích phản hồi của hành khách về dịch vụ đường sắt bằng các thuật toán học máy.

## Chương 1. Tổng quan
1.1. Lý do chọn đề tài  
1.2. Mục tiêu  
1.3. Đối tượng và phạm vi  
1.4. Phương pháp thực hiện  
1.5. Cấu trúc báo cáo  

## Chương 2. Cơ sở lý thuyết
2.1. Phân tích cảm xúc và phân loại văn bản  
2.2. Tiền xử lý văn bản tiếng Việt  
2.3. TF-IDF  
2.4. Multinomial Naive Bayes  
2.5. Linear Support Vector Machine  
2.6. Các chỉ số: Accuracy, Precision, Recall, Macro-F1 và ma trận nhầm lẫn  
2.7. Công cụ: Python, pandas, scikit-learn, Streamlit  

## Chương 3. Phân tích và thiết kế hệ thống
3.1. Yêu cầu chức năng  
3.2. Nguồn dữ liệu, điều kiện sử dụng và cách gán nhãn  
3.3. Làm sạch dữ liệu và phân bố nhãn  
3.4. Quy trình xử lý: dữ liệu → chia tập → TF-IDF → mô hình → đánh giá  
3.5. Thiết kế giao diện và luồng sử dụng  

## Chương 4. Cài đặt và thực nghiệm
4.1. Môi trường phát triển  
4.2. Cấu trúc mã nguồn  
4.3. Cài đặt TF-IDF và hai mô hình  
4.4. Thiết lập thực nghiệm, random_state và tỷ lệ chia dữ liệu  
4.5. Bảng kết quả của hai mô hình  
4.6. Phân tích ma trận nhầm lẫn và các trường hợp dự đoán sai  
4.7. Ảnh chụp màn hình demo  

## Chương 5. Đánh giá và hướng phát triển
5.1. Kết quả đạt được  
5.2. Hạn chế của dữ liệu và mô hình  
5.3. Phần tự phát triển so với tài liệu/mã nguồn tham khảo  
5.4. Hướng phát triển: dữ liệu đường sắt lớn hơn, phân tích theo khía cạnh, mô hình ngôn ngữ tiếng Việt  

## Kết luận
Tóm tắt công việc đã làm, kết quả thực nghiệm thực tế và hướng mở rộng.

## Tài liệu tham khảo
- Tài liệu bài báo về UIT-VSFC nếu có sử dụng.
- Tài liệu chính thức scikit-learn về TfidfVectorizer, MultinomialNB, LinearSVC và metrics.
- Tài liệu chính thức Streamlit về file uploader và download button.
- Nguồn dữ liệu phản hồi đường sắt thực tế và ngày truy cập.

## Bảng kết quả cần điền sau khi chạy thực nghiệm
| Mô hình | Accuracy | Macro Precision | Macro Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | Điền kết quả thật | Điền | Điền | Điền |
| Linear SVM | Điền kết quả thật | Điền | Điền | Điền |

Không điền số liệu dự đoán hoặc lấy kết quả từ dữ liệu minh họa làm kết quả chính thức.
