from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
VALID_LABELS = {"tich_cuc", "tieu_cuc", "trung_tinh"}
LABEL_DISPLAY = {
    "tich_cuc": "Tích cực",
    "tieu_cuc": "Tiêu cực",
    "trung_tinh": "Trung tính",
}

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân tích cảm xúc phản hồi.")
    parser.add_argument("--data", default="data/raw/feedback_raw.csv",
                        help="CSV có 2 cột text,label. Mặc định: data/raw/feedback_raw.csv")
    parser.add_argument("--output", default="models", help="Thư mục lưu mô hình và báo cáo")
    args = parser.parse_args()

    data_path = (ROOT / args.data).resolve()
    output_dir = (ROOT / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {data_path}. Hãy tạo data/feedback.csv từ dữ liệu đã gán nhãn. "
            "File data/sample_demo.csv chỉ là dữ liệu minh họa để kiểm tra chương trình."
        )

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV phải có đúng tên cột bắt buộc: text và label.")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""]
    df = df[df["label"].isin(VALID_LABELS)]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if df["label"].nunique() < 2:
        raise ValueError("Dữ liệu cần ít nhất 2 nhóm nhãn hợp lệ.")
    if df["label"].value_counts().min() < 2:
        raise ValueError("Mỗi nhóm nhãn cần ít nhất 2 câu để chia tập dữ liệu.")

    print(f"Số phản hồi sau làm sạch: {len(df)}")
    print("Phân bố nhãn:")
    print(df["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)),
            ("classifier", MultinomialNB())
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42))
        ])
    }

    results = {}
    best_name = None
    best_f1 = -1
    best_model = None
    labels = ["tieu_cuc", "trung_tinh", "tich_cuc"]

    for name, model in models.items():
        print(f"\nĐang huấn luyện: {name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="macro", zero_division=0
        )
        results[name] = {
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "classification_report": classification_report(
                y_test, predictions, labels=labels, target_names=[
                    "Tiêu cực", "Trung tính", "Tích cực"
                ], output_dict=True, zero_division=0
            )
        }
        print(f"Accuracy: {accuracy:.4f} | Macro F1: {f1:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model
            best_predictions = predictions

    joblib.dump(best_model, output_dir / "best_model.joblib")
    (output_dir / "model_name.txt").write_text(best_name, encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    display_labels = ["Tiêu cực", "Trung tính", "Tích cực"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Ma trận nhầm lẫn - {best_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    # Lưu danh sách câu kiểm tra và dự đoán để phục vụ phân tích lỗi trong báo cáo.
    error_df = pd.DataFrame({
        "text": X_test.reset_index(drop=True),
        "label_thuc_te": y_test.reset_index(drop=True),
        "du_doan": pd.Series(best_predictions)
    })
    error_df = error_df[error_df["label_thuc_te"] != error_df["du_doan"]]
    error_df.to_csv(output_dir / "prediction_errors.csv", index=False, encoding="utf-8-sig")

    print("\nHoàn thành.")
    print(f"Mô hình được chọn theo Macro F1: {best_name}")
    print(f"Đã lưu mô hình và kết quả tại: {output_dir}")
    print("Lưu ý: kết quả chỉ có ý nghĩa khi dữ liệu đầu vào có nguồn gốc và nhãn đáng tin cậy.")

if __name__ == "__main__":
    main()
