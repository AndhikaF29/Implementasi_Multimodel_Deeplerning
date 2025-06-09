"""
ANALISIS PREDIKSI KELANGSUNGAN HIDUP PENUMPANG TITANIC MENGGUNAKAN GRU
=====================================================================

Dataset: Titanic Dataset
Model: Gated Recurrent Unit (GRU)
Tujuan: Memprediksi apakah seorang penumpang selamat atau tidak

Kolom Dataset:
- PassengerId: ID unik penumpang
- Survived: Target (0 = tidak selamat, 1 = selamat)
- Pclass: Kelas tiket (1, 2, 3)
- Name: Nama penumpang
- Sex: Jenis kelamin
- Age: Umur
- SibSp: Jumlah saudara/pasangan di kapal
- Parch: Jumlah orangtua/anak di kapal
- Ticket: Nomor tiket
- Fare: Harga tiket
- Cabin: Nomor kabin
- Embarked: Pelabuhan keberangkatan
"""

# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seed untuk reprodusibilitas
np.random.seed(42)
tf.random.set_seed(42)

class TitanicGRUPredictor:
    """
    Kelas untuk prediksi kelangsungan hidup penumpang Titanic menggunakan GRU
    """
    
    def __init__(self, data_path):
        """
        Inisialisasi dengan path data
        
        Args:
            data_path (str): Path ke file dataset Titanic
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.history = None
        
    def load_and_explore_data(self):
        """
        Memuat dan melakukan eksplorasi awal data
        """
        print("=" * 60)
        print("TAHAP 1: MEMUAT DAN MENGEKSPLORASI DATA")
        print("=" * 60)
        
        # Memuat data
        self.data = pd.read_csv(self.data_path)
        
        print(f"Bentuk data: {self.data.shape}")
        print(f"Jumlah baris: {self.data.shape[0]}")
        print(f"Jumlah kolom: {self.data.shape[1]}")
        print("\nInfo dataset:")
        print(self.data.info())
        
        print("\n5 baris pertama:")
        print(self.data.head())
        
        print("\nStatistik deskriptif:")
        print(self.data.describe())
        
        # Cek missing values
        print("\nMissing values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Distribusi target variable
        print("\nDistribusi target variable (Survived):")
        survival_counts = self.data['Survived'].value_counts()
        print(survival_counts)
        print(f"Persentase selamat: {survival_counts[1]/len(self.data)*100:.2f}%")
        print(f"Persentase tidak selamat: {survival_counts[0]/len(self.data)*100:.2f}%")
        
    def visualize_data(self):
        """
        Membuat visualisasi untuk memahami data
        """
        print("\n" + "=" * 60)
        print("TAHAP 2: VISUALISASI DATA")
        print("=" * 60)
        
        # Setup plotting
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Analisis Dataset Titanic', fontsize=16, fontweight='bold')
        
        # 1. Distribusi survival
        self.data['Survived'].value_counts().plot(kind='bar', ax=axes[0,0], color=['red', 'green'])
        axes[0,0].set_title('Distribusi Kelangsungan Hidup')
        axes[0,0].set_xlabel('Survived (0: Tidak, 1: Ya)')
        axes[0,0].set_ylabel('Jumlah')
        axes[0,0].set_xticklabels(['Tidak Selamat', 'Selamat'], rotation=0)
        
        # 2. Survival berdasarkan gender
        survival_by_sex = pd.crosstab(self.data['Sex'], self.data['Survived'])
        survival_by_sex.plot(kind='bar', ax=axes[0,1], color=['red', 'green'])
        axes[0,1].set_title('Kelangsungan Hidup berdasarkan Jenis Kelamin')
        axes[0,1].set_xlabel('Jenis Kelamin')
        axes[0,1].set_ylabel('Jumlah')
        axes[0,1].legend(['Tidak Selamat', 'Selamat'])
        
        # 3. Survival berdasarkan kelas
        survival_by_class = pd.crosstab(self.data['Pclass'], self.data['Survived'])
        survival_by_class.plot(kind='bar', ax=axes[0,2], color=['red', 'green'])
        axes[0,2].set_title('Kelangsungan Hidup berdasarkan Kelas')
        axes[0,2].set_xlabel('Kelas')
        axes[0,2].set_ylabel('Jumlah')
        axes[0,2].legend(['Tidak Selamat', 'Selamat'])
        
        # 4. Distribusi umur
        self.data['Age'].hist(bins=30, ax=axes[1,0], alpha=0.7, color='skyblue')
        axes[1,0].set_title('Distribusi Umur')
        axes[1,0].set_xlabel('Umur')
        axes[1,0].set_ylabel('Frekuensi')
        
        # 5. Distribusi harga tiket
        self.data['Fare'].hist(bins=30, ax=axes[1,1], alpha=0.7, color='orange')
        axes[1,1].set_title('Distribusi Harga Tiket')
        axes[1,1].set_xlabel('Harga Tiket')
        axes[1,1].set_ylabel('Frekuensi')
        
        # 6. Heatmap korelasi
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1,2], 
                   fmt='.2f', cbar_kws={'shrink': .8})
        axes[1,2].set_title('Matriks Korelasi')
        
        plt.tight_layout()
        plt.show()
        
    def preprocess_data(self):
        """
        Melakukan preprocessing data untuk model GRU
        """
        print("\n" + "=" * 60)
        print("TAHAP 3: PREPROCESSING DATA")
        print("=" * 60)
        
        # Copy data untuk preprocessing
        df = self.data.copy()
        
        # 1. Menangani missing values
        print("Menangani missing values...")
        
        # Age: isi dengan median berdasarkan Pclass dan Sex
        df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Embarked: isi dengan modus
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
        # Fare: isi dengan median berdasarkan Pclass
        df['Fare'] = df.groupby('Pclass')['Fare'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Drop kolom yang tidak diperlukan
        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        df = df.drop(columns=columns_to_drop)
        
        print(f"Kolom yang digunakan: {list(df.columns)}")
        
        # 2. Feature Engineering
        print("Melakukan feature engineering...")
        
        # Membuat fitur baru
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 100], 
                               labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        
        # 3. Encoding categorical variables
        print("Melakukan encoding categorical variables...")
        
        categorical_cols = ['Sex', 'Embarked', 'AgeGroup', 'FareGroup']
        
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # 4. Memisahkan features dan target
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        print(f"Bentuk features: {X.shape}")
        print(f"Bentuk target: {y.shape}")
        print(f"Fitur yang digunakan: {list(X.columns)}")
        
        # 5. Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 6. Scaling features
        print("Melakukan scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # 7. Reshape untuk GRU (samples, timesteps, features)
        # Untuk GRU, kita perlu menambahkan dimensi timesteps
        # Kita akan treat setiap feature sebagai timestep
        self.X_train_gru = self.X_train_scaled.reshape(
            self.X_train_scaled.shape[0], self.X_train_scaled.shape[1], 1
        )
        self.X_test_gru = self.X_test_scaled.reshape(
            self.X_test_scaled.shape[0], self.X_test_scaled.shape[1], 1
        )
        
        print(f"Bentuk data training untuk GRU: {self.X_train_gru.shape}")
        print(f"Bentuk data testing untuk GRU: {self.X_test_gru.shape}")
        
        print("Preprocessing selesai!")
        
    def build_gru_model(self):
        """
        Membangun model GRU untuk prediksi
        """
        print("\n" + "=" * 60)
        print("TAHAP 4: MEMBANGUN MODEL GRU")
        print("=" * 60)
        
        # Arsitektur GRU
        self.model = Sequential([
            # Layer GRU pertama dengan return sequences
            GRU(64, return_sequences=True, input_shape=(self.X_train_gru.shape[1], 1),
                dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Layer GRU kedua
            GRU(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Fully connected layers
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            # Output layer untuk binary classification
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        print("Arsitektur Model GRU:")
        self.model.summary()
        
        # Visualisasi arsitektur (opsional)
        try:
            tf.keras.utils.plot_model(
                self.model, 
                to_file='model_architecture.png', 
                show_shapes=True, 
                show_layer_names=True
            )
            print("Diagram arsitektur model disimpan sebagai 'model_architecture.png'")
        except:
            print("Tidak dapat membuat diagram arsitektur model")
            
    def train_model(self):
        """
        Melatih model GRU
        """
        print("\n" + "=" * 60)
        print("TAHAP 5: MELATIH MODEL")
        print("=" * 60)
        
        # Callbacks untuk training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        print("Memulai training model...")
        print(f"Training samples: {len(self.X_train_gru)}")
        print(f"Validation samples: {len(self.X_test_gru)}")
        
        # Training model
        self.history = self.model.fit(
            self.X_train_gru, self.y_train,
            validation_data=(self.X_test_gru, self.y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training selesai!")
        
    def plot_training_history(self):
        """
        Membuat plot riwayat training
        """
        print("\n" + "=" * 60)
        print("TAHAP 6: VISUALISASI TRAINING")
        print("=" * 60)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy During Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print final metrics
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        print(f"Akurasi Training Akhir: {final_train_acc:.4f}")
        print(f"Akurasi Validasi Akhir: {final_val_acc:.4f}")
        print(f"Loss Training Akhir: {final_train_loss:.4f}")
        print(f"Loss Validasi Akhir: {final_val_loss:.4f}")
        
    def evaluate_model(self):
        """
        Evaluasi performa model
        """
        print("\n" + "=" * 60)
        print("TAHAP 7: EVALUASI MODEL")
        print("=" * 60)
        
        # Prediksi
        y_pred_proba = self.model.predict(self.X_test_gru)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Akurasi
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Akurasi Model: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nLaporan Klasifikasi:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Tidak Selamat', 'Selamat']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 4))
        
        # Plot confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Tidak Selamat', 'Selamat'],
                   yticklabels=['Tidak Selamat', 'Selamat'])
        plt.title('Confusion Matrix')
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        
        # Plot prediction distribution
        plt.subplot(1, 2, 2)
        plt.hist(y_pred_proba, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        plt.title('Distribusi Probabilitas Prediksi')
        plt.xlabel('Probabilitas Selamat')
        plt.ylabel('Frekuensi')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return accuracy, y_pred, y_pred_proba
        
    def predict_new_data(self, new_data):
        """
        Memprediksi data baru
        
        Args:
            new_data (dict): Dictionary berisi data penumpang baru
            
        Returns:
            tuple: (probabilitas, prediksi)
        """
        # Convert ke DataFrame
        df_new = pd.DataFrame([new_data])
        
        # Preprocessing yang sama seperti training data
        # Feature engineering
        df_new['FamilySize'] = df_new['SibSp'] + df_new['Parch'] + 1
        df_new['IsAlone'] = (df_new['FamilySize'] == 1).astype(int)
        
        # Age group
        df_new['AgeGroup'] = pd.cut(df_new['Age'], bins=[0, 12, 18, 35, 50, 100], 
                                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # Fare group (gunakan quantile dari training data)
        # Untuk simplifikasi, kita buat manual berdasarkan nilai training
        if df_new['Fare'].iloc[0] <= 7.91:
            df_new['FareGroup'] = 'Low'
        elif df_new['Fare'].iloc[0] <= 14.45:
            df_new['FareGroup'] = 'Medium'
        elif df_new['Fare'].iloc[0] <= 31.0:
            df_new['FareGroup'] = 'High'
        else:
            df_new['FareGroup'] = 'Very High'
        
        # Encoding
        categorical_cols = ['Sex', 'Embarked', 'AgeGroup', 'FareGroup']
        for col in categorical_cols:
            if col in self.label_encoders:
                try:
                    df_new[col] = self.label_encoders[col].transform(df_new[col].astype(str))
                except:
                    # Jika nilai tidak dikenal, gunakan nilai yang paling sering
                    df_new[col] = 0
        
        # Pilih kolom yang sama seperti training
        expected_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                        'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup']
        df_new = df_new[expected_cols]
        
        # Scaling
        X_new_scaled = self.scaler.transform(df_new)
        
        # Reshape untuk GRU
        X_new_gru = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)
        
        # Prediksi
        probability = self.model.predict(X_new_gru)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        return probability, prediction
        
    def run_complete_analysis(self):
        """
        Menjalankan analisis lengkap dari awal hingga akhir
        """
        print("🚢 ANALISIS PREDIKSI KELANGSUNGAN HIDUP PENUMPANG TITANIC 🚢")
        print("Menggunakan Gated Recurrent Unit (GRU)")
        print("=" * 80)
        
        # Tahap 1: Load dan eksplorasi data
        self.load_and_explore_data()
        
        # Tahap 2: Visualisasi
        self.visualize_data()
        
        # Tahap 3: Preprocessing
        self.preprocess_data()
        
        # Tahap 4: Build model
        self.build_gru_model()
        
        # Tahap 5: Training
        self.train_model()
        
        # Tahap 6: Plot training history
        self.plot_training_history()
        
        # Tahap 7: Evaluasi
        accuracy, y_pred, y_pred_proba = self.evaluate_model()
        
        # Tahap 8: Contoh prediksi
        print("\n" + "=" * 60)
        print("TAHAP 8: CONTOH PREDIKSI DATA BARU")
        print("=" * 60)
        
        # Contoh data penumpang baru
        contoh_penumpang = {
            'Pclass': 3,
            'Sex': 'male',
            'Age': 22,
            'SibSp': 1,
            'Parch': 0,
            'Fare': 7.25,
            'Embarked': 'S'
        }
        
        probability, prediction = self.predict_new_data(contoh_penumpang)
        
        print("Contoh Prediksi:")
        print(f"Data penumpang: {contoh_penumpang}")
        print(f"Probabilitas selamat: {probability:.4f} ({probability*100:.2f}%)")
        print(f"Prediksi: {'Selamat' if prediction == 1 else 'Tidak Selamat'}")
        
        # Summary
        print("\n" + "=" * 60)
        print("RINGKASAN HASIL")
        print("=" * 60)
        print(f"✅ Dataset berhasil diproses: {len(self.data)} sampel")
        print(f"✅ Model GRU berhasil dilatih")
        print(f"✅ Akurasi model: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ Model siap digunakan untuk prediksi")
        
        return self.model, accuracy

# Fungsi utama untuk menjalankan analisis
def main():
    """
    Fungsi utama untuk menjalankan analisis lengkap
    """
    # Path ke dataset
    data_path = "Titanic-Dataset.csv"
    
    # Inisialisasi predictor
    predictor = TitanicGRUPredictor(data_path)
    
    # Jalankan analisis lengkap
    model, accuracy = predictor.run_complete_analysis()
    
    return predictor, model, accuracy

# Jalankan jika file ini dieksekusi langsung
if __name__ == "__main__":
    predictor, model, accuracy = main()
    
    print("\n🎉 Analisis selesai!")
    print("Anda dapat menggunakan 'predictor' untuk prediksi data baru.")
    print("Contoh penggunaan:")
    print("probability, prediction = predictor.predict_new_data({")
    print("    'Pclass': 1, 'Sex': 'female', 'Age': 25, 'SibSp': 0,")
    print("    'Parch': 0, 'Fare': 50, 'Embarked': 'C'")
    print("})")