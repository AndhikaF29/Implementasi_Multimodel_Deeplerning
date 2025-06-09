"""
🩺 Diabetes Classification using GRU Neural Network
===================================================

Sistem klasifikasi diabetes menggunakan Gated Recurrent Unit (GRU) 
dengan preprocessing data yang comprehensive dan evaluasi model yang mendalam.

Author: Deep Learning Team
Dataset: Pima Indians Diabetes Database
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, roc_auc_score, f1_score, 
                           precision_score, recall_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (GRU, Dense, Dropout, BatchNormalization, 
                                   Input, Bidirectional, GlobalMaxPooling1D,
                                   GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import os

# Konfigurasi environment
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

class DiabetesClassifierGRU:
    """
    🩺 Kelas utama untuk klasifikasi diabetes menggunakan GRU
    
    Class ini menghandle seluruh pipeline dari data loading hingga evaluasi model,
    dengan menggunakan arsitektur GRU yang dioptimasi untuk data tabular.
    """
    
    def __init__(self, data_path: str = None):
        """
        Inisialisasi classifier
        
        Args:
            data_path (str): Path ke file dataset diabetes.csv
        """
        self.data_path = data_path or r'd:\Kuliah\Tugas kuliah SM6\deeplerning\TUBES\diabetes\diabetes.csv'
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        print("🩺 Diabetes Classifier GRU initialized!")
        print(f"📁 Dataset path: {self.data_path}")
    
    def load_and_explore_data(self):
        """
        📊 Load dataset dan melakukan eksplorasi data awal
        """
        print("\n" + "="*60)
        print("📊 LOADING DAN EKSPLORASI DATA")
        print("="*60)
        
        try:
            # Load dataset
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset berhasil dimuat!")
            print(f"📋 Shape dataset: {self.df.shape}")
            
            # Informasi dasar dataset
            print(f"\n📈 Informasi Dataset:")
            print(f"   - Jumlah pasien: {len(self.df)}")
            print(f"   - Jumlah fitur: {len(self.df.columns) - 1}")
            print(f"   - Target variable: Outcome (0=Non-Diabetes, 1=Diabetes)")
            
            # Statistik target variable
            outcome_counts = self.df['Outcome'].value_counts()
            print(f"\n🎯 Distribusi Target:")
            print(f"   - Non-Diabetes (0): {outcome_counts[0]} ({outcome_counts[0]/len(self.df)*100:.1f}%)")
            print(f"   - Diabetes (1): {outcome_counts[1]} ({outcome_counts[1]/len(self.df)*100:.1f}%)")
            
            # Check missing values
            missing_data = self.df.isnull().sum()
            print(f"\n❓ Missing Values:")
            if missing_data.sum() == 0:
                print("   ✅ Tidak ada missing values")
            else:
                for col, missing in missing_data[missing_data > 0].items():
                    print(f"   - {col}: {missing} ({missing/len(self.df)*100:.1f}%)")
            
            # Deteksi nilai 0 yang mencurigakan
            suspicious_zeros = {}
            zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in zero_cols:
                zeros = (self.df[col] == 0).sum()
                if zeros > 0:
                    suspicious_zeros[col] = zeros
            
            if suspicious_zeros:
                print(f"\n⚠️ Nilai 0 yang Mencurigakan:")
                for col, zeros in suspicious_zeros.items():
                    print(f"   - {col}: {zeros} samples ({zeros/len(self.df)*100:.1f}%)")
            
            return self.df
            
        except FileNotFoundError:
            print(f"❌ Error: File {self.data_path} tidak ditemukan!")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def visualize_data_distribution(self):
        """
        📊 Visualisasi distribusi data
        """
        print("\n" + "="*60)
        print("📊 VISUALISASI DISTRIBUSI DATA")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Distribution of target variable
        plt.subplot(3, 4, 1)
        outcome_counts = self.df['Outcome'].value_counts()
        colors = ['#3498db', '#e74c3c']
        plt.pie(outcome_counts.values, labels=['Non-Diabetes', 'Diabetes'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('🎯 Distribusi Target Variable', fontsize=12, fontweight='bold')
        
        # 2-9. Distribution of features
        feature_cols = self.df.columns[:-1]  # Exclude 'Outcome'
        for i, col in enumerate(feature_cols, 2):
            plt.subplot(3, 4, i)
            
            # Histogram with KDE
            self.df[self.df['Outcome']==0][col].hist(alpha=0.7, label='Non-Diabetes', 
                                                   color='#3498db', bins=20)
            self.df[self.df['Outcome']==1][col].hist(alpha=0.7, label='Diabetes', 
                                                   color='#e74c3c', bins=20)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.title(f'📊 {col} Distribution')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('diabetes_data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=0.5)
        plt.title('🔗 Correlation Matrix - Diabetes Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('diabetes_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualisasi selesai! Grafik disimpan sebagai PNG files.")
    
    def preprocess_data(self):
        """
        🔧 Preprocessing data untuk training
        """
        print("\n" + "="*60)
        print("🔧 PREPROCESSING DATA")
        print("="*60)
        
        # Handle nilai 0 yang mencurigakan dengan median replacement
        df_processed = self.df.copy()
        
        # Kolom yang tidak boleh 0 (kecuali benar-benar 0)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        print("🔄 Menangani nilai 0 yang mencurigakan...")
        for col in zero_cols:
            zeros_count = (df_processed[col] == 0).sum()
            if zeros_count > 0:
                # Replace 0 dengan median dari non-zero values
                median_val = df_processed[df_processed[col] > 0][col].median()
                df_processed.loc[df_processed[col] == 0, col] = median_val
                print(f"   - {col}: {zeros_count} nilai 0 diganti dengan median ({median_val:.2f})")
        
        # Feature Engineering
        print("\n🎯 Feature Engineering...")
        
        # 1. BMI Categories
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 0  # Underweight
            elif bmi < 25:
                return 1  # Normal
            elif bmi < 30:
                return 2  # Overweight
            else:
                return 3  # Obese
        
        df_processed['BMI_Category'] = df_processed['BMI'].apply(categorize_bmi)
        
        # 2. Age Groups
        def categorize_age(age):
            if age < 30:
                return 0  # Young
            elif age < 50:
                return 1  # Middle
            else:
                return 2  # Senior
        
        df_processed['Age_Group'] = df_processed['Age'].apply(categorize_age)
        
        # 3. Glucose Level Categories
        def categorize_glucose(glucose):
            if glucose < 100:
                return 0  # Normal
            elif glucose < 126:
                return 1  # Prediabetes
            else:
                return 2  # Diabetes range
        
        df_processed['Glucose_Category'] = df_processed['Glucose'].apply(categorize_glucose)
        
        # 4. Blood Pressure Categories
        def categorize_bp(bp):
            if bp < 80:
                return 0  # Low
            elif bp < 120:
                return 1  # Normal
            elif bp < 140:
                return 2  # High normal
            else:
                return 3  # High
        
        df_processed['BP_Category'] = df_processed['BloodPressure'].apply(categorize_bp)
        
        # 5. Insulin Resistance Indicator
        df_processed['Insulin_Resistance'] = (df_processed['Insulin'] > 166).astype(int)
        
        # 6. Risk Score (composite feature)
        df_processed['Risk_Score'] = (
            df_processed['Glucose'] * 0.3 +
            df_processed['BMI'] * 0.2 +
            df_processed['Age'] * 0.1 +
            df_processed['DiabetesPedigreeFunction'] * 100 * 0.4
        )
        
        print(f"   ✅ Menambahkan 6 fitur engineered")
        print(f"   📊 Total fitur sekarang: {len(df_processed.columns) - 1}")
        
        # Separate features and target
        feature_columns = [col for col in df_processed.columns if col != 'Outcome']
        X = df_processed[feature_columns].values
        y = df_processed['Outcome'].values
        
        print(f"\n📋 Shape final dataset:")
        print(f"   - Features (X): {X.shape}")
        print(f"   - Target (y): {y.shape}")
        
        # Split data: 70% train, 15% validation, 15% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 0.15/0.85
        )
        
        print(f"\n📊 Data Split:")
        print(f"   - Training: {self.X_train.shape[0]} samples ({self.X_train.shape[0]/len(df_processed)*100:.1f}%)")
        print(f"   - Validation: {self.X_val.shape[0]} samples ({self.X_val.shape[0]/len(df_processed)*100:.1f}%)")
        print(f"   - Test: {self.X_test.shape[0]} samples ({self.X_test.shape[0]/len(df_processed)*100:.1f}%)")
        
        # Scale features
        print("\n⚖️ Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Reshape untuk GRU (samples, time_steps, features)
        # Kita akan treat setiap feature sebagai time step
        self.X_train_gru = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], self.X_train_scaled.shape[1], 1)
        self.X_val_gru = self.X_val_scaled.reshape(self.X_val_scaled.shape[0], self.X_val_scaled.shape[1], 1)
        self.X_test_gru = self.X_test_scaled.reshape(self.X_test_scaled.shape[0], self.X_test_scaled.shape[1], 1)
        
        print(f"   ✅ Features scaled menggunakan StandardScaler")
        print(f"   🔄 Reshaped untuk GRU: {self.X_train_gru.shape}")
        
        # Class distribution check
        train_dist = np.bincount(self.y_train)
        val_dist = np.bincount(self.y_val)
        test_dist = np.bincount(self.y_test)
        
        print(f"\n🎯 Distribusi kelas setelah split:")
        print(f"   - Training: Non-Diabetes={train_dist[0]}, Diabetes={train_dist[1]}")
        print(f"   - Validation: Non-Diabetes={val_dist[0]}, Diabetes={val_dist[1]}")
        print(f"   - Test: Non-Diabetes={test_dist[0]}, Diabetes={test_dist[1]}")
        
        return True
    
    def build_gru_model(self):
        """
        🏗️ Membangun arsitektur model GRU untuk klasifikasi diabetes
        """
        print("\n" + "="*60)
        print("🏗️ MEMBANGUN MODEL GRU")
        print("="*60)
        
        # Calculate class weights untuk handling imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        self.class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"⚖️ Class weights: {self.class_weight_dict}")
        
        # Model architecture
        input_shape = (self.X_train_gru.shape[1], self.X_train_gru.shape[2])
        print(f"📥 Input shape: {input_shape}")
        
        self.model = Sequential([
            # Input layer
            Input(shape=input_shape, name='input_layer'),
            
            # First Bidirectional GRU layer
            Bidirectional(
                GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
                name='bidirectional_gru_1'
            ),
            BatchNormalization(name='batch_norm_1'),
            
            # Second Bidirectional GRU layer  
            Bidirectional(
                GRU(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
                name='bidirectional_gru_2'
            ),
            BatchNormalization(name='batch_norm_2'),
            
            # Global pooling layers
            GlobalMaxPooling1D(name='global_max_pool'),
            
            # Dense layers
            Dense(64, activation='relu', kernel_regularizer=l2(0.01), name='dense_1'),
            Dropout(0.4, name='dropout_1'),
            BatchNormalization(name='batch_norm_3'),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='dense_2'),
            Dropout(0.3, name='dropout_2'),
            BatchNormalization(name='batch_norm_4'),
            
            Dense(16, activation='relu', kernel_regularizer=l2(0.01), name='dense_3'),
            Dropout(0.2, name='dropout_3'),
            
            # Output layer
            Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Model summary
        print("\n🏗️ Arsitektur Model:")
        print("-" * 50)
        self.model.summary()
        
        # Plot model architecture
        try:
            plot_model(self.model, to_file='diabetes_gru_model.png', show_shapes=True, 
                      show_layer_names=True, rankdir='TB', dpi=300)
            print("\n📊 Model architecture diagram saved as 'diabetes_gru_model.png'")
        except:
            print("\n⚠️ Could not save model diagram (graphviz not installed)")
        
        print(f"\n📊 Model Summary:")
        print(f"   - Total parameters: {self.model.count_params():,}")
        print(f"   - Trainable parameters: {sum([np.prod(p.shape) for p in self.model.trainable_weights]):,}")
        print(f"   - Optimizer: Adam (lr=0.001)")
        print(f"   - Loss function: Binary Crossentropy")
        print(f"   - Metrics: Accuracy, Precision, Recall")
        
        return self.model
    
    def train_model(self, epochs=100, batch_size=32, verbose=1):
        """
        🚀 Training model GRU
        
        Args:
            epochs (int): Jumlah epoch training
            batch_size (int): Ukuran batch
            verbose (int): Verbosity level
        """
        print("\n" + "="*60)
        print("🚀 TRAINING MODEL GRU")
        print("="*60)
          # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                'best_diabetes_gru_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            )
        ]
        
        print(f"📋 Training Configuration:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Early stopping patience: 15")
        print(f"   - Learning rate reduction patience: 8")
        print(f"   - Class weights: {self.class_weight_dict}")
        
        print(f"\n🚀 Starting training...")
        print("-" * 50)
        
        # Train model
        self.history = self.model.fit(
            self.X_train_gru, self.y_train,
            validation_data=(self.X_val_gru, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=self.class_weight_dict,
            verbose=verbose
        )
        
        print(f"\n✅ Training completed!")
        print(f"📊 Training epochs: {len(self.history.history['loss'])}")
        
        return self.history
    
    def plot_training_history(self):
        """
        📈 Visualisasi history training
        """
        print("\n📈 Plotting training history...")
        
        if self.history is None:
            print("❌ No training history found. Train the model first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss', color='#e74c3c')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', color='#3498db')
        axes[0, 0].set_title('📉 Model Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy', color='#e74c3c')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='#3498db')
        axes[0, 1].set_title('📈 Model Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision', color='#e74c3c')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision', color='#3498db')
        axes[1, 0].set_title('🎯 Model Precision', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall', color='#e74c3c')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall', color='#3498db')
        axes[1, 1].set_title('🔍 Model Recall', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('diabetes_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training history plots saved as 'diabetes_training_history.png'")
    
    def evaluate_model(self):
        """
        📊 Evaluasi comprehensive model
        """
        print("\n" + "="*60)
        print("📊 EVALUASI MODEL")
        print("="*60)
        
        # Predictions
        train_pred = self.model.predict(self.X_train_gru, verbose=0)
        val_pred = self.model.predict(self.X_val_gru, verbose=0)
        test_pred = self.model.predict(self.X_test_gru, verbose=0)
        
        # Convert probabilities to binary predictions
        train_pred_binary = (train_pred > 0.5).astype(int).flatten()
        val_pred_binary = (val_pred > 0.5).astype(int).flatten()
        test_pred_binary = (test_pred > 0.5).astype(int).flatten()
        
        # Calculate metrics
        results = {}
        
        # Training metrics
        results['train_accuracy'] = accuracy_score(self.y_train, train_pred_binary)
        results['train_precision'] = precision_score(self.y_train, train_pred_binary)
        results['train_recall'] = recall_score(self.y_train, train_pred_binary)
        results['train_f1'] = f1_score(self.y_train, train_pred_binary)
        results['train_auc'] = roc_auc_score(self.y_train, train_pred.flatten())
        
        # Validation metrics
        results['val_accuracy'] = accuracy_score(self.y_val, val_pred_binary)
        results['val_precision'] = precision_score(self.y_val, val_pred_binary)
        results['val_recall'] = recall_score(self.y_val, val_pred_binary)
        results['val_f1'] = f1_score(self.y_val, val_pred_binary)
        results['val_auc'] = roc_auc_score(self.y_val, val_pred.flatten())
        
        # Test metrics
        results['test_accuracy'] = accuracy_score(self.y_test, test_pred_binary)
        results['test_precision'] = precision_score(self.y_test, test_pred_binary)
        results['test_recall'] = recall_score(self.y_test, test_pred_binary)
        results['test_f1'] = f1_score(self.y_test, test_pred_binary)
        results['test_auc'] = roc_auc_score(self.y_test, test_pred.flatten())
        
        # Print results
        print("📊 HASIL EVALUASI MODEL:")
        print("-" * 60)
        print(f"{'Metric':<15} {'Train':<8} {'Validation':<12} {'Test':<8}")
        print("-" * 60)
        print(f"{'Accuracy':<15} {results['train_accuracy']:<8.4f} {results['val_accuracy']:<12.4f} {results['test_accuracy']:<8.4f}")
        print(f"{'Precision':<15} {results['train_precision']:<8.4f} {results['val_precision']:<12.4f} {results['test_precision']:<8.4f}")
        print(f"{'Recall':<15} {results['train_recall']:<8.4f} {results['val_recall']:<12.4f} {results['test_recall']:<8.4f}")
        print(f"{'F1-Score':<15} {results['train_f1']:<8.4f} {results['val_f1']:<12.4f} {results['test_f1']:<8.4f}")
        print(f"{'AUC-ROC':<15} {results['train_auc']:<8.4f} {results['val_auc']:<12.4f} {results['test_auc']:<8.4f}")
        print("-" * 60)
        
        # Classification Report
        print(f"\n📋 CLASSIFICATION REPORT (Test Set):")
        print("-" * 50)
        print(classification_report(self.y_test, test_pred_binary, 
                                  target_names=['Non-Diabetes', 'Diabetes']))
        
        # Confusion Matrix Visualization
        self.plot_confusion_matrix(self.y_test, test_pred_binary)
        
        # ROC Curve
        self.plot_roc_curve(self.y_test, test_pred.flatten())
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        📊 Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Diabetes', 'Diabetes'],
                   yticklabels=['Non-Diabetes', 'Diabetes'])
        plt.title('🎯 Confusion Matrix - Diabetes Classification', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1%})', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig('diabetes_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Confusion matrix saved as 'diabetes_confusion_matrix.png'")
    
    def plot_roc_curve(self, y_true, y_scores):
        """
        📈 Plot ROC curve
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#e74c3c', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('📈 ROC Curve - Diabetes Classification', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('diabetes_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ ROC curve saved as 'diabetes_roc_curve.png'")
    
    def predict_diabetes(self, patient_data):
        """
        🩺 Prediksi diabetes untuk pasien baru
        
        Args:
            patient_data (dict): Data pasien dengan key sesuai feature names
            
        Returns:
            dict: Hasil prediksi dengan probabilitas dan interpretasi
        """
        # Convert to DataFrame
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = pd.DataFrame(patient_data)
        
        # Ensure all required features are present
        required_features = self.feature_names
        for feature in required_features:
            if feature not in patient_df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Apply same preprocessing
        patient_processed = patient_df[required_features].copy()
        
        # Handle suspicious zeros (same as training)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in patient_processed.columns:
                median_val = self.df[self.df[col] > 0][col].median()
                patient_processed.loc[patient_processed[col] == 0, col] = median_val
        
        # Feature engineering (same as training)
        patient_processed['BMI_Category'] = patient_processed['BMI'].apply(
            lambda x: 0 if x < 18.5 else 1 if x < 25 else 2 if x < 30 else 3
        )
        patient_processed['Age_Group'] = patient_processed['Age'].apply(
            lambda x: 0 if x < 30 else 1 if x < 50 else 2
        )
        patient_processed['Glucose_Category'] = patient_processed['Glucose'].apply(
            lambda x: 0 if x < 100 else 1 if x < 126 else 2
        )
        patient_processed['BP_Category'] = patient_processed['BloodPressure'].apply(
            lambda x: 0 if x < 80 else 1 if x < 120 else 2 if x < 140 else 3
        )
        patient_processed['Insulin_Resistance'] = (patient_processed['Insulin'] > 166).astype(int)
        patient_processed['Risk_Score'] = (
            patient_processed['Glucose'] * 0.3 +
            patient_processed['BMI'] * 0.2 +
            patient_processed['Age'] * 0.1 +
            patient_processed['DiabetesPedigreeFunction'] * 100 * 0.4
        )
        
        # Scale and reshape
        patient_scaled = self.scaler.transform(patient_processed.values)
        patient_gru = patient_scaled.reshape(patient_scaled.shape[0], patient_scaled.shape[1], 1)
        
        # Predict
        probability = self.model.predict(patient_gru, verbose=0)[0][0]
        prediction = "Diabetes" if probability > 0.5 else "Non-Diabetes"
        confidence = probability if probability > 0.5 else 1 - probability
        
        # Risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        return {
            'prediction': prediction,
            'probability': float(probability),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'recommendation': self._get_recommendation(probability, patient_data)
        }
    
    def _get_recommendation(self, probability, patient_data):
        """
        💡 Memberikan rekomendasi berdasarkan hasil prediksi
        """
        if probability < 0.3:
            return "Risiko diabetes rendah. Pertahankan gaya hidup sehat."
        elif probability < 0.7:
            return "Risiko diabetes sedang. Disarankan konsultasi dengan dokter dan pemeriksaan lebih lanjut."
        else:
            return "Risiko diabetes tinggi. Segera konsultasi dengan dokter untuk pemeriksaan dan penanganan lebih lanjut."

def main():
    """
    🚀 Fungsi utama untuk menjalankan seluruh pipeline
    """
    print("🩺" + "="*59)
    print("🩺 DIABETES CLASSIFICATION USING GRU NEURAL NETWORK")
    print("🩺" + "="*59)
    
    # Initialize classifier
    classifier = DiabetesClassifierGRU()
    
    # 1. Load and explore data
    df = classifier.load_and_explore_data()
    if df is None:
        return
    
    # 2. Visualize data distribution
    classifier.visualize_data_distribution()
    
    # 3. Preprocess data
    success = classifier.preprocess_data()
    if not success:
        print("❌ Preprocessing failed!")
        return
    
    # 4. Build model
    model = classifier.build_gru_model()
    
    # 5. Train model
    history = classifier.train_model(epochs=50, batch_size=32)
    
    # 6. Plot training history
    classifier.plot_training_history()
    
    # 7. Evaluate model
    results = classifier.evaluate_model()
    
    # 8. Test with sample predictions
    print("\n" + "="*60)
    print("🩺 TESTING DENGAN CONTOH PASIEN")
    print("="*60)
    
    # Sample patients
    sample_patients = [
        {
            'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72, 'SkinThickness': 35,
            'Insulin': 0, 'BMI': 33.6, 'DiabetesPedigreeFunction': 0.627, 'Age': 50
        },
        {
            'Pregnancies': 1, 'Glucose': 85, 'BloodPressure': 66, 'SkinThickness': 29,
            'Insulin': 0, 'BMI': 26.6, 'DiabetesPedigreeFunction': 0.351, 'Age': 31
        },
        {
            'Pregnancies': 8, 'Glucose': 183, 'BloodPressure': 64, 'SkinThickness': 0,
            'Insulin': 0, 'BMI': 23.3, 'DiabetesPedigreeFunction': 0.672, 'Age': 32
        }
    ]
    
    for i, patient in enumerate(sample_patients, 1):
        result = classifier.predict_diabetes(patient)
        print(f"\n👤 Pasien {i}:")
        print(f"   📊 Data: Glucose={patient['Glucose']}, BMI={patient['BMI']}, Age={patient['Age']}")
        print(f"   🎯 Prediksi: {result['prediction']}")
        print(f"   📈 Probabilitas: {result['probability']:.4f}")
        print(f"   💯 Confidence: {result['confidence']:.4f}")
        print(f"   ⚠️ Risk Level: {result['risk_level']}")
        print(f"   💡 Rekomendasi: {result['recommendation']}")
    
    print(f"\n✅ ANALISIS COMPLETED SUCCESSFULLY!")
    print(f"🎯 Final Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"📈 Final Test AUC Score: {results['test_auc']:.4f}")

if __name__ == "__main__":
    main()