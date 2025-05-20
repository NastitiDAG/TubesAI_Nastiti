import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QFrame,
    QPushButton, QFileDialog, QLineEdit, QMessageBox, QScrollArea,
    QStackedWidget, QHBoxLayout, QGroupBox, QFormLayout, QTextEdit, 
    QCheckBox, QComboBox, QTableWidget, QHeaderView, QTableWidgetItem
)
from PyQt6.QtGui import QFont, QDoubleValidator, QTextCursor
from PyQt6.QtCore import Qt
from sklearn.model_selection import cross_val_score
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, confusion_matrix
)

class DashboardPage(QWidget):      
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input Dataset
        label_input = QLabel("Input Dataset")
        layout.addWidget(label_input)

        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Pilih file dataset CSV...")
        self.file_input.setReadOnly(True)
        file_layout.addWidget(self.file_input)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)

        layout.addLayout(file_layout)

        # Tombol Prediksi
        self.predict_btn = QPushButton("Prediksi")
        self.predict_btn.clicked.connect(self.prediksi)
        layout.addWidget(self.predict_btn)

        # Tabel Hasil Prediksi
        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.setLayout(layout)

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih File CSV", "", "CSV Files (*.csv)")
        if file_name:
            self.file_input.setText(file_name)

    def prediksi(self):
        file_path = self.file_input.text()
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "File Tidak Ditemukan", "Silakan pilih file CSV yang valid.")
            return

        try:
            # Load data CSV
            df = pd.read_csv(file_path)

            # Load model dan encoder yang sudah kamu simpan
            model = joblib.load("model_cuaca.joblib")
            encoder = joblib.load("model_cuaca_encoder.joblib")

            # Jika kolom 'Label' ada di data, hapus supaya hanya input fitur yang dipakai
            if 'Label' in df.columns:
                df = df.drop(columns=['Label'])

            # Pastikan data fitur sesuai yang model harapkan
            # Jika ada fitur kategorikal yang perlu encoding, sesuaikan dengan preprocessing yang kamu pakai waktu training
            # Contoh sederhana jika semua fitur numerik, langsung ke prediksi

            # Cek apakah model expect fitur yang sama (jumlah kolom dan nama)
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
                # Reorder atau pilih fitur yang sesuai model
                df = df[expected_features]

            # Prediksi label
            y_pred = model.predict(df)

            # Ubah label hasil prediksi ke nama asli (jika pakai encoder)
            if encoder:
                y_label = encoder.inverse_transform(y_pred)
            else:
                y_label = y_pred

            # Tambahkan kolom hasil prediksi ke DataFrame
            df['Label_Prediksi'] = y_label

            # Tampilkan hasil di tabel
            self.load_table(df)

            QMessageBox.information(self, "Sukses", "Prediksi berhasil dilakukan dan ditampilkan.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Terjadi kesalahan saat prediksi:\n{str(e)}")

    def load_table(self, df):
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns.tolist())

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, item)




class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Training Model Cuaca")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # File Input Section
        file_group = QGroupBox("Input Dataset")
        file_layout = QVBoxLayout()
        
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Pilih file dataset CSV...")
        self.file_input.setReadOnly(True)
        
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.load_file)
        
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_input)
        file_row.addWidget(browse_button)
        
        file_layout.addLayout(file_row)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Training Parameters Section
        params_group = QGroupBox("Parameter Training")
        params_layout = QFormLayout()
        
        # self.model_name_input = QLineEdit()
        # self.model_name_input.setPlaceholderText("Contoh: Model_Cuaca_v1")
        
        self.validation_input = QLineEdit()
        self.validation_input.setPlaceholderText("0.2")
        self.validation_input.setValidator(QDoubleValidator(0.1, 0.9, 2))
        
        # params_layout.addRow("Nama Model:", self.model_name_input)
        params_layout.addRow("Rasio Validasi:", self.validation_input)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Buttons Section
        button_layout = QHBoxLayout()
        
        self.labeling_button = QPushButton("Labeling Data")
        self.labeling_button.clicked.connect(self.label_data)
        button_layout.addWidget(self.labeling_button)
        
        self.split_button = QPushButton("SplitData")
        self.split_button.clicked.connect(self.split_data)
        button_layout.addWidget(self.split_button)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        button_layout.addWidget(self.train_button)
        
        layout.addLayout(button_layout)

        # Results Container
        results_group = QGroupBox("Hasil Training")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Hasil training akan ditampilkan di sini...")
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                color: black;
            }
        """)
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Back button
        self.back_button = QPushButton("Kembali ke Menu Utama")
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

        self.setLayout(layout)
        self.dataset_path = None

    def split_data(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Peringatan", "Silakan pilih file dataset terlebih dahulu.")
            return

        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal membaca file: {str(e)}")
            return

        try:
            test_size = float(self.validation_input.text())
        except ValueError:
            QMessageBox.warning(self, "Peringatan", "Masukkan rasio validasi yang valid (misal: 0.2).")
            return

        # Pisahkan fitur dan target
        X = df.drop(columns=['Label'])
        y = df['Label']

        # Encode fitur kategorikal (jika ada)
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        # Encode target jika perlu (opsional, jika y juga string)
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        try:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

            train_df.to_csv("data_training.csv", index=False)
            test_df.to_csv("data_testing.csv", index=False)

            QMessageBox.information(
                self,
                "Split Data Berhasil",
                f"Data berhasil dibagi:\n\n"
                f"- Data Training: {len(train_df)} baris\n"
                f"- Data Testing: {len(test_df)} baris\n\n"
                f"Disimpan sebagai:\n- data_training.csv\n- data_testing.csv"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Terjadi kesalahan saat split data:\n{str(e)}")

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Pilih File Dataset", 
            "", 
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.dataset_path = file_path
            self.file_input.setText(file_path)
            
            # Preview dataset
            try:
                df = pd.read_csv(file_path)
                self.results_text.setPlainText(
                    f"Dataset berhasil dimuat:\n"
                    f"Path: {file_path}\n"
                    f"Jumlah baris: {len(df)}\n"
                    f"Kolom: {', '.join(df.columns)}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal memuat dataset: {str(e)}")

    def label_data(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Peringatan", "Silakan pilih file dataset terlebih dahulu.")
            return

        try:
            self.results_text.clear()
            self.results_text.append("Memulai proses labeling...")
            QApplication.processEvents()

            df = pd.read_csv(self.dataset_path)
            self.results_text.append(f"\nDataset berhasil dimuat. Jumlah data: {len(df)}")

            required_columns = ['suhu', 'kelembapan', 'lux', 'hujan']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing)}")

            self.results_text.append("\nMemproses label berdasarkan kondisi cuaca...")
            QApplication.processEvents()

            # Logika labeling
            conditions = [
                (df['hujan'] == 1),  # Hujan
                (df['lux'] < 1000),  # Mendung
                ((df['lux'] >= 1000) & (df['lux'] <= 7000)),  # Berawan
                (df['lux'] > 7000) & (df['suhu'] > 30),  # Cerah
            ]

            choices = [3, 2, 1, 0]  # Hujan=3, Mendung=2, Berawan=1, Cerah=0
            df['Label'] = np.select(conditions, choices, default=1)  # Default Berawan jika tidak cocok

            # Simpan hasil
            save_path = self.dataset_path.replace('.csv', '_labeled.csv')
            df.to_csv(save_path, index=False)

            self.results_text.append("\n=== Hasil Labeling ===")
            self.results_text.append(f"File tersimpan di: {save_path}")
            self.results_text.append("\nDistribusi Label:")

            label_map = {0: 'Cerah', 1: 'Berawan', 2: 'Mendung', 3: 'Hujan'}
            label_counts = df['Label'].value_counts().sort_index()
            for kode, count in label_counts.items():
                nama_label = label_map.get(kode, f"Label {kode}")
                self.results_text.append(f"{nama_label} ({kode}): {count} data ({count/len(df)*100:.1f}%)")

            self.results_text.append("\nSample Data (5 baris pertama):")
            self.results_text.append(df.head().to_string(index=False))

            cursor = self.results_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.results_text.setTextCursor(cursor)

            QMessageBox.information(
                self,
                "Labeling Berhasil",
                f"Data berhasil dilabeli dan disimpan di:\n{save_path}"
            )

        except Exception as e:
            self.results_text.append(f"\nError: {str(e)}")
            QMessageBox.critical(self, "Error", f"Gagal memproses dataset: {str(e)}")

    from sklearn.preprocessing import LabelEncoder

    def train_model(self):
        try:
            # ===== VALIDASI FILE =====
            if not os.path.exists("data_training.csv"):
                raise FileNotFoundError("File data_training.csv tidak ditemukan. Lakukan split data terlebih dahulu.")
            if not os.path.exists("data_testing.csv"):
                raise FileNotFoundError("File data_testing.csv tidak ditemukan. Lakukan split data terlebih dahulu.")

            self.results_text.clear()
            self.results_text.setTextColor(Qt.GlobalColor.black)
            self.results_text.append("Memulai proses training model...")
            QApplication.processEvents()

            # ===== LOAD DATA =====
            train_df = pd.read_csv("data_training.csv")
            test_df = pd.read_csv("data_testing.csv")

            # Validasi kolom Label
            for df, name in [(train_df, "training"), (test_df, "testing")]:
                if 'Label' not in df.columns:
                    raise ValueError(f"Kolom 'Label' tidak ditemukan dalam file {name}.")

            # Pisahkan fitur dan target
            X_train = train_df.drop(columns=['Label'])
            y_train = train_df['Label']
            X_test = test_df.drop(columns=['Label'])
            y_test = test_df['Label']

            # ===== ENCODE FITUR & LABEL =====
            self.results_text.append("\nMengonversi data kategorikal...")
            label_encoder = LabelEncoder()
            encoders = {}

            # Encode fitur kategorikal
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    le = LabelEncoder()
                    X_train[col] = le.fit_transform(X_train[col].astype(str))
                    X_test[col] = le.transform(X_test[col].astype(str))  # Gunakan encoder yang sama
                    encoders[col] = le

            # Encode label
            if y_train.dtype == 'object':
                y_train = label_encoder.fit_transform(y_train.astype(str))
                y_test = label_encoder.transform(y_test.astype(str))

            # Simpan encoder
            joblib.dump(encoders, "feature_encoders.joblib")
            joblib.dump(label_encoder, "label_encoder.joblib")

            # ===== VALIDASI DATA =====
            self.results_text.append("\nValidasi data...")
            for X, name in [(X_train, "Training"), (X_test, "Testing")]:
                if X.isnull().values.any():
                    raise ValueError(f"Data {name} mengandung nilai kosong (NaN).")
                if not all(dtype.kind in 'fi' for dtype in X.dtypes):
                    raise ValueError(f"Semua fitur di data {name} harus numerik.")

            if len(set(y_train)) > 10:
                raise ValueError("Jumlah kelas label terlalu banyak (>10).")
            if len(X_train) < 20 or len(X_test) < 10:
                raise ValueError("Jumlah data terlalu sedikit (minimal 20 training & 10 testing).")

            # ===== TRAINING MODEL =====
            self.results_text.append("\nMelatih model Decision Tree...")
            model = DecisionTreeClassifier(
                criterion='entropy',
                max_depth=3,  # Lebih konservatif untuk hindari overfitting
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            model.fit(X_train, y_train)

            # ===== EVALUASI =====
            def evaluate(X, y, dataset_name):
                y_pred = model.predict(X)
                self.results_text.append(f"\n=== Hasil Evaluasi ({dataset_name}) ===")
                self.results_text.append(f"Akurasi       : {accuracy_score(y, y_pred):.4f}")
                self.results_text.append(f"Presisi       : {precision_score(y, y_pred, average='weighted', zero_division=0):.4f}")
                self.results_text.append(f"Recall        : {recall_score(y, y_pred, average='weighted', zero_division=0):.4f}")
                self.results_text.append(f"F1 Score      : {f1_score(y, y_pred, average='weighted', zero_division=0):.4f}")

            evaluate(X_train, y_train, "Training Set")
            evaluate(X_test, y_test, "Testing Set")

            # Cross Validation
            self.results_text.append("\n=== Cross Validation (5-fold) ===")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            self.results_text.append(f"Akurasi CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

            # ===== FEATURE IMPORTANCE =====
            self.results_text.append("\nFeature Importance:")
            for name, importance in zip(X_train.columns, model.feature_importances_):
                self.results_text.append(f"- {name}: {importance:.4f}")

            # ===== SIMPAN MODEL =====
            model_path = "model_cuaca.joblib"
            joblib.dump(model, model_path)
            self.results_text.append(f"\nModel berhasil disimpan di: {model_path}")

            # Tampilkan peringatan jika ada overfitting
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            if train_acc - test_acc > 0.2:
                self.results_text.setTextColor(Qt.GlobalColor.darkYellow)
                self.results_text.append("\nPERINGATAN: Model mungkin overfitting (akurasi training jauh lebih tinggi dari testing)!")

            # Scroll ke atas dan tampilkan notifikasi
            self.results_text.moveCursor(QTextCursor.MoveOperation.Start)
            QMessageBox.information(
                self,
                "Training Selesai",
                f"Proses training berhasil.\n"
                f"Akurasi Training: {train_acc:.2%}\n"
                f"Akurasi Testing: {test_acc:.2%}\n"
                f"Akurasi CV: {cv_scores.mean():.2%}"
            )

        except Exception as e:
            self.results_text.setTextColor(Qt.GlobalColor.red)
            self.results_text.append(f"\nERROR: {str(e)}")
            QMessageBox.critical(self, "Error", f"Gagal melatih model:\n{str(e)}")
        finally:
            self.results_text.setTextColor(Qt.GlobalColor.black)


    def go_back(self):
        parent = self.parentWidget()
        if parent and isinstance(parent, QStackedWidget):
            parent.setCurrentIndex(0)


class EvaluationPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # 1. Judul Halaman
        self.title_label = QLabel("Evaluasi Model")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.label_encoder = None

         # ===== INPUT SECTION =====
        input_group = QGroupBox("Input Evaluasi")
        input_layout = QVBoxLayout()

        # Model path input
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Pilih model .joblib...")
        self.model_input.setReadOnly(True)
        model_btn = QPushButton("Browse Model")
        model_btn.clicked.connect(self.browse_model)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_input)
        model_layout.addWidget(model_btn)
        input_layout.addLayout(model_layout)

        # Model path input
        self.encoder_input = QLineEdit()
        self.encoder_input.setPlaceholderText("Pilih model encoder .joblib...")
        self.encoder_input.setReadOnly(True)
        encoder_btn = QPushButton("Browse Model encoder")
        encoder_btn.clicked.connect(self.browse_encoder)

        encoder_layout = QHBoxLayout()
        encoder_layout.addWidget(self.encoder_input)
        encoder_layout.addWidget(encoder_btn)
        input_layout.addLayout(encoder_layout)

        # Data path input
        self.data_input = QLineEdit()
        self.data_input.setPlaceholderText("Pilih data_testing.csv...")
        self.data_input.setReadOnly(True)
        data_btn = QPushButton("Browse Data")
        data_btn.clicked.connect(self.browse_data)

        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_input)
        data_layout.addWidget(data_btn)
        input_layout.addLayout(data_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # ===== BUTTONS FOR PLOTS =====
        plot_buttons = QHBoxLayout()

        self.btn_confusion = QPushButton("Confusion Matrix")
        self.btn_confusion.clicked.connect(self.show_confusion_matrix)
        plot_buttons.addWidget(self.btn_confusion)

        self.btn_feature_importance = QPushButton("Feature Importance Plot")
        self.btn_feature_importance.clicked.connect(self.show_feature_importance)
        plot_buttons.addWidget(self.btn_feature_importance)

        self.btn_learning_curve = QPushButton("Learning Curve")
        self.btn_learning_curve.clicked.connect(self.show_learning_curve)
        plot_buttons.addWidget(self.btn_learning_curve)

        layout.addLayout(plot_buttons)

        # ===== RESULTS CONTAINER =====
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Hasil evaluasi akan ditampilkan di sini...")
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                color: black;
            }
        """)
        layout.addWidget(self.results_text)

        # ===== EVALUATE BUTTON =====
        eval_btn = QPushButton("Evaluasi")
        eval_btn.clicked.connect(self.evaluate_model)
        layout.addWidget(eval_btn)

        self.setLayout(layout)

        # Variables
        self.model = None
        self.X_test = None
        self.y_test = None
        self.columns = []

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Model .joblib", "", "Model Files (*.joblib)")
        if path:
            self.model_input.setText(path)
            self.model = joblib.load(path)

    def browse_encoder(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Label Encoder .joblib", "", "Joblib Files (*.joblib)")
        if path:
            self.encoder_input.setText(path)
            self.label_encoder = joblib.load(path)

    def browse_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Data Testing", "", "CSV Files (*.csv)")
        if path:
            self.data_input.setText(path)
            df = pd.read_csv(path)
            if 'Label' not in df.columns:
                QMessageBox.critical(self, "Error", "Kolom 'Label' tidak ditemukan dalam file.")
                return
            self.X_test = df.drop(columns=['Label'])
            self.y_test = df['Label']
            self.columns = self.X_test.columns.tolist()

    def evaluate_model(self):
        try:
            if self.model is None or self.X_test is None:
                raise ValueError("Model atau data belum dipilih.")

            # Encode label jika masih dalam bentuk string
            if self.label_encoder:
                if isinstance(self.y_test.iloc[0], str):
                    self.y_test = self.label_encoder.transform(self.y_test)

            y_pred = self.model.predict(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)

            self.results_text.clear()
            self.results_text.append("=== Hasil Evaluasi Model ===")
            self.results_text.append(f"Akurasi       : {acc:.4f}")
            self.results_text.append(f"Presisi       : {prec:.4f}")
            self.results_text.append(f"Recall        : {rec:.4f}")
            self.results_text.append(f"F1 Score      : {f1:.4f}")
            self.results_text.append(f"MAE           : {mae:.4f}")
            self.results_text.append(f"MSE           : {mse:.4f}")

        except Exception as e:
            QMessageBox.critical(self, "Error Evaluasi", str(e))

    def show_confusion_matrix(self):
        try:
            if self.model is None or self.X_test is None:
                raise ValueError("Model dan data testing belum dimuat.")
            y_pred = self.model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def show_feature_importance(self):
        try:
            if self.model is None or not hasattr(self.model, 'feature_importances_'):
                raise ValueError("Model tidak memiliki feature importance.")
            importances = self.model.feature_importances_
            plt.figure(figsize=(8, 5))
            plt.barh(self.columns, importances, color="skyblue")
            plt.xlabel("Importance")
            plt.title("Feature Importance")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def show_learning_curve(self):
        try:
            if self.model is None or self.X_test is None:
                raise ValueError("Model dan data testing belum dimuat.")
            from sklearn.model_selection import learning_curve
            from sklearn.tree import DecisionTreeClassifier

            # Re-train untuk learning curve karena butuh training set
            X = pd.concat([self.X_test, self.X_test], ignore_index=True)  # gunakan data dobel
            y = pd.concat([self.y_test, self.y_test], ignore_index=True)
            train_sizes, train_scores, test_scores = learning_curve(
                DecisionTreeClassifier(), X, y,
                train_sizes=np.linspace(0.1, 1.0, 5),
                cv=5, scoring='accuracy'
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_scores_mean, label="Training Score")
            plt.plot(train_sizes, test_scores_mean, label="Validation Score")
            plt.title("Learning Curve")
            plt.xlabel("Training Set Size")
            plt.ylabel("Accuracy")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikasi Prediksi Cuaca")
        self.setGeometry(100, 100, 800, 600)

        # Navigasi & StackedWidget
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        nav_layout = QHBoxLayout()

        btn_dashboard = QPushButton("Dashboard")
        btn_dashboard.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        btn_training = QPushButton("Training")
        btn_training.clicked.connect(lambda: self.pages.setCurrentIndex(1))
        btn_eval = QPushButton("Evaluasi")
        btn_eval.clicked.connect(lambda: self.pages.setCurrentIndex(2))
        

        for btn in [btn_dashboard, btn_training, btn_eval]:
            btn.setStyleSheet("background-color: #2c2c2c; color: white; padding: 10px;")
            nav_layout.addWidget(btn)

        self.pages = QStackedWidget()
        self.pages.addWidget(DashboardPage())
        self.pages.addWidget(TrainingPage())
        self.pages.addWidget(EvaluationPage())
       

        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.pages)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.setStyleSheet("background-color: #121212; color: white;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()  # Harus MainWindow bukan QMainWindow biasa
    window.show()
    sys.exit(app.exec())
