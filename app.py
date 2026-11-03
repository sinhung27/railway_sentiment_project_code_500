from pathlib import Path
import io
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
NAME_PATH = ROOT / "models" / "model_name.txt"

LABEL_DISPLAY = {
    "tich_cuc": "Tích cực",
    "tieu_cuc": "Tiêu cực",
    "trung_tinh": "Trung tính",
}
LABEL_EMOJI = {"tich_cuc": "😊", "tieu_cuc": "☹️", "trung_tinh": "😐"}

st.set_page_config(page_title="Phân tích phản hồi đường sắt", page_icon="🚆", layout="wide")
st.title("🚆 Phân tích cảm xúc phản hồi hành khách đường sắt")
st.caption("Đồ án môn Trí tuệ nhân tạo — TF-IDF kết hợp học máy")

if not MODEL_PATH.exists():
    st.error("Chưa tìm thấy mô hình đã huấn luyện.")
    st.markdown("""
    **Cách khắc phục**
    1. Chuẩn bị `data/feedback.csv` có hai cột `text` và `label`.
    2. Mở Command Prompt tại thư mục dự án và chạy: `python train.py`
    3. Chạy lại ứng dụng: `streamlit run app.py`

    `data/sample_demo.csv` chỉ chứa câu minh họa để kiểm tra chương trình, không dùng làm kết quả thực nghiệm chính thức.
    """)
    st.stop()

model = joblib.load(MODEL_PATH)
model_name = NAME_PATH.read_text(encoding="utf-8").strip() if NAME_PATH.exists() else "Mô hình đã huấn luyện"

with st.sidebar:
    st.header("Thông tin mô hình")
    st.write(f"**Mô hình đang dùng:** {model_name}")
    st.write("Đầu vào: văn bản tiếng Việt")
    st.write("Đầu ra: tích cực / tiêu cực / trung tính")
    st.info("Kết quả là dự đoán của mô hình, không phải đánh giá tuyệt đối về chất lượng dịch vụ.")

tab1, tab2 = st.tabs(["Phân tích một phản hồi", "Phân tích tệp CSV"])

with tab1:
    st.subheader("Nhập phản hồi")
    text = st.text_area(
        "Nội dung phản hồi",
        placeholder="Ví dụ: Nhân viên nhiệt tình, toa tàu sạch sẽ và tàu đến đúng giờ.",
        height=130
    )
    if st.button("Phân tích cảm xúc", type="primary"):
        if not text.strip():
            st.warning("Bạn hãy nhập nội dung phản hồi trước.")
        else:
            prediction = model.predict([text.strip()])[0]
            st.metric("Kết quả dự đoán", f"{LABEL_EMOJI.get(prediction, '')} {LABEL_DISPLAY.get(prediction, prediction)}")
            st.write("**Nội dung đã phân tích:**")
            st.write(text.strip())

with tab2:
    st.subheader("Tải lên tệp CSV")
    st.write("Tệp cần có cột phản hồi. Chương trình tự nhận diện một trong các tên cột: `text`, `feedback`, `comment`, `noi_dung`.")
    uploaded = st.file_uploader("Chọn tệp CSV (UTF-8)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, encoding="utf-8-sig")
            candidates = ["text", "feedback", "comment", "noi_dung", "phan_hoi", "review"]
            text_col = next((c for c in candidates if c in df.columns), None)
            if text_col is None:
                st.error("Không tìm thấy cột phản hồi. Hãy đặt tên cột là text, feedback, comment hoặc noi_dung.")
            else:
                df = df.dropna(subset=[text_col]).copy()
                df[text_col] = df[text_col].astype(str).str.strip()
                df = df[df[text_col] != ""]
                if df.empty:
                    st.warning("Tệp không có phản hồi hợp lệ.")
                else:
                    df["cam_xuc_du_doan"] = model.predict(df[text_col].tolist())
                    df["nhan_cam_xuc"] = df["cam_xuc_du_doan"].map(LABEL_DISPLAY)
                    st.success(f"Đã phân tích {len(df)} phản hồi.")
                    counts = df["nhan_cam_xuc"].value_counts().reindex(
                        ["Tích cực", "Trung tính", "Tiêu cực"], fill_value=0
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tích cực", int(counts["Tích cực"]))
                    c2.metric("Trung tính", int(counts["Trung tính"]))
                    c3.metric("Tiêu cực", int(counts["Tiêu cực"]))
                    st.bar_chart(counts)
                    st.dataframe(df, use_container_width=True)
                    output = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Tải kết quả CSV",
                        data=output,
                        file_name="ket_qua_phan_tich_cam_xuc.csv",
                        mime="text/csv"
                    )
        except Exception as exc:
            st.error(f"Không đọc được tệp CSV: {exc}")

st.divider()
st.caption("Đồ án sinh viên. Không dùng kết quả tự động làm căn cứ duy nhất cho quyết định vận hành hoặc đánh giá chính thức.")
