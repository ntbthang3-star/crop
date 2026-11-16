# AgriSuggest – Streamlit

## Chạy trên Windows
1. Đặt `Dataset_Machine_Learning.csv` cùng thư mục với `app.py` (hoặc tải CSV ở thanh bên).
2. Mở Terminal/CMD tại thư mục dự án.
3. Cài thư viện: `pip install -r requirements.txt`
4. Chạy: `streamlit run app.py`

## Dataset
Cần có 7 cột đầu vào: `N`, `P`, `K`, `ph`, `temperature`, `humidity`, `rainfall` và cột nhãn `label` (chấp nhận tên `crop`/`Crop`). Đơn vị N/P/K phải theo dataset gốc.

## Tính năng
Dashboard; dự đoán 7 thông số; Top-3; so sánh hai mô hình; Accuracy, Macro Precision/Recall/F1; Confusion Matrix; Classification Report; khám phá/lọc/tải dataset; dự đoán hàng loạt CSV; lịch sử dự đoán trong phiên; cảnh báo dữ liệu ngoài phạm vi Train.

## Lưu ý về mô hình tự triển khai
Bản mẫu tự thực hiện Bootstrap, huấn luyện nhiều cây và bỏ phiếu; các cây quyết định cơ sở dùng `DecisionTreeClassifier` của Scikit-learn. Nếu yêu cầu môn học bắt buộc tự cài đặt cả thuật toán xây dựng cây, cần thay phần cây cơ sở bằng code tự viết. `predict_proba` không phải xác suất cây chắc chắn sinh trưởng tốt.
