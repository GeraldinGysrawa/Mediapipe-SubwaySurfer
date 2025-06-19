import json
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load gesture data
GESTURE_JSON = 'head_gestures/head_gestures.json'

# Fungsi untuk load data
def load_data():
    with open(GESTURE_JSON, 'r') as f:
        gesture_db = json.load(f)
    X = []
    y = []
    for gesture in gesture_db:
        pose = gesture['head_pose']
        features = [pose['yaw'], pose['pitch'], pose['roll']]
        X.append(features)
        y.append(gesture['name'])
    return np.array(X), np.array(y)

def train_svm_model():
    X, y = load_data()
    print(f"Jumlah data: {len(X)}")
    print(f"Fitur: {X.shape[1]} (yaw, pitch, roll)")

    # Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode label
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Cross-validation (5-fold)
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_reports = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_encoded), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Latih SVM
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        fold_reports.append((fold, acc, report))
        print(f"Fold {fold} - Akurasi: {acc:.2f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\nAkurasi Rata-rata: {mean_acc:.2f} ± {std_acc:.2f}\n")

    print("Laporan Klasifikasi per Fold:")
    for fold, acc, report in fold_reports:
        print(f"\nFold {fold} (Akurasi: {acc:.2f}):")
        for gesture, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"  {gesture}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-Score={metrics['f1-score']:.2f}")

    # Latih ulang model di seluruh data
    final_model = SVC(kernel='rbf', probability=True, random_state=42)
    final_model.fit(X_scaled, y_encoded)

    # Simpan model, scaler, dan label encoder
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(final_model, 'models/gesture_svm.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    print("\nModel SVM, scaler, dan label encoder berhasil disimpan di folder 'models/'.")

if __name__ == '__main__':
    train_svm_model() 