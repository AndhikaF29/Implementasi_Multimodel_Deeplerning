
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Embedding, GRU, LSTM, Dense, Dropout, 
                                   BatchNormalization, Bidirectional, GlobalMaxPooling1D,
                                   Input, Concatenate, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Set random seeds untuk reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Stopwords Indonesia yang diperluas
INDONESIAN_STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'adalah', 'akan',
    'dalam', 'sebagai', 'oleh', 'tidak', 'atau', 'dapat', 'telah', 'juga', 'ini',
    'itu', 'saya', 'kami', 'kita', 'anda', 'mereka', 'dia', 'ia', 'ada', 'sudah',
    'harus', 'bisa', 'hanya', 'lebih', 'sangat', 'setelah', 'sebelum', 'karena',
    'jika', 'bila', 'maka', 'tetapi', 'namun', 'tapi', 'seperti', 'antara',
    'hingga', 'sampai', 'selama', 'tanpa', 'melalui', 'terhadap', 'tentang',
    'about', 'above', 'after', 'again', 'all', 'and', 'any', 'are', 'as', 'at',
    'be', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few',
    'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'if', 'in',
    'into', 'is', 'it', 'its', 'itself', 'me', 'more', 'most', 'my', 'myself',
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should',
    'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
    'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

# Kata-kata yang sering muncul di spam
SPAM_INDICATORS = {
    'menang', 'hadiah', 'gratis', 'bonus', 'promo', 'diskon', 'untung', 'profit',
    'investasi', 'modal', 'uang', 'rupiah', 'dollar', 'bisnis', 'peluang',
    'klik', 'link', 'daftar', 'gabung', 'buruan', 'terbatas', 'sekarang',
    'jangan', 'sampai', 'ketinggalan', 'tawaran', 'khusus', 'eksklusif',
    'segera', 'cepat', 'mudah', 'instant', 'langsung', 'otomatis',
    'guarantee', 'guaranteed', 'free', 'win', 'winner', 'prize', 'money',
    'cash', 'earn', 'profit', 'income', 'investment', 'business', 'opportunity',
    'click', 'urgent', 'limited', 'offer', 'deal', 'sale', 'discount'
}

def focal_loss(gamma=2., alpha=0.25):
    """Focal Loss untuk mengatasi class imbalance"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        
        return K.mean(weight * cross_entropy)
    
    return focal_loss_fixed

class SuperEnhancedEmailClassifier:
    def __init__(self, max_words=15000, max_length=200):
        """Inisialisasi classifier super enhanced"""
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.class_weights = None
        self.tfidf_vectorizer = None
        
    def load_and_analyze_data(self):
        """Load dan analisis mendalam dataset"""
        print("📁 Loading dan menganalisis dataset...")
        
        possible_paths = [
            r'd:\Kuliah\Tugas kuliah SM6\deeplerning\TUBES\email\email_spam_indo.csv',
            r'email_spam_indo.csv',
            r'.\email_spam_indo.csv',
            r'..\email_spam_indo.csv',
            r'email\email_spam_indo.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    print(f"🔍 Mencoba path: {path}")
                    df = pd.read_csv(path)
                    print(f"✅ Dataset berhasil di-load dari: {path}")
                    print(f"📊 Dataset berisi {len(df)} email")
                    print(f"📊 Kolom dataset: {list(df.columns)}")
                    
                    # Analisis mendalam
                    print("\n📈 Distribusi Label:")
                    label_counts = df['Kategori'].value_counts()
                    print(label_counts)
                    print(f"📊 Rasio Ham:Spam = {label_counts['ham']}:{label_counts['spam']}")
                    
                    # Analisis panjang teks
                    print("\n📏 Analisis Panjang Teks:")
                    df['text_length'] = df['Pesan'].str.len()
                    print(f"Ham - Mean length: {df[df['Kategori']=='ham']['text_length'].mean():.1f}")
                    print(f"Spam - Mean length: {df[df['Kategori']=='spam']['text_length'].mean():.1f}")
                    
                    return df
                    
                except Exception as e:
                    print(f"❌ Error loading dataset dari {path}: {e}")
                    continue
        
        print("❌ Dataset tidak ditemukan!")
        return None
    
    def super_advanced_text_cleaning(self, text):
        """Pembersihan teks super canggih dengan feature preservation"""
        if pd.isna(text):
            return ""
        
        original_text = str(text)
        text = original_text.lower()
        
        # Preserve important spam indicators sebelum cleaning
        spam_features = []
        for indicator in SPAM_INDICATORS:
            if indicator in text:
                spam_features.append(indicator)
        
        # Advanced cleaning
        text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)  # Replace URL dengan token
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)              # Replace email dengan token
        text = re.sub(r'<[^>]+>', '', text)                     # Hapus HTML tags
        text = re.sub(r'\d+', ' NUMBER ', text)                 # Replace angka dengan token
        
        # Preserve exclamation marks dan capital patterns
        exclamation_count = text.count('!')
        caps_ratio = sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1)
        
        # Clean punctuation tapi preserve beberapa
        text = re.sub(r'[^\w\s!]', ' ', text)
        text = re.sub(r'!+', ' EXCLAMATION ', text)
        
        # Normalize repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Tokenize
        words = text.split()
        
        # Filter words
        processed_words = []
        for word in words:
            # Keep special tokens
            if word in ['URL', 'EMAIL', 'NUMBER', 'EXCLAMATION']:
                processed_words.append(word)
            # Keep spam indicators
            elif word in SPAM_INDICATORS:
                processed_words.append(word)
            # Keep normal words with good length
            elif 2 <= len(word) <= 20 and word not in INDONESIAN_STOPWORDS:
                processed_words.append(word)
        
        # Add spam features back
        processed_words.extend(spam_features)
        
        # Add feature tokens
        if exclamation_count > 2:
            processed_words.append('MANY_EXCLAMATION')
        if caps_ratio > 0.3:
            processed_words.append('MANY_CAPS')
        
        result = ' '.join(processed_words)
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def extract_advanced_features(self, df):
        """Ekstraksi fitur lanjutan"""
        print("🔍 Mengekstrak fitur lanjutan...")
        
        features = pd.DataFrame()
        
        # Basic text features
        features['text_length'] = df['Pesan'].str.len()
        features['word_count'] = df['clean_text'].str.split().str.len()
        features['unique_word_ratio'] = df['clean_text'].apply(
            lambda x: len(set(x.split())) / max(len(x.split()), 1) if x else 0
        )
        
        # Spam indicator features
        features['spam_word_count'] = df['Pesan'].apply(
            lambda x: sum(1 for word in SPAM_INDICATORS if word in str(x).lower())
        )
        features['spam_word_ratio'] = features['spam_word_count'] / features['word_count'].replace(0, 1)
        
        # Punctuation features
        features['exclamation_count'] = df['Pesan'].str.count('!')
        features['question_count'] = df['Pesan'].str.count('\?')
        features['caps_ratio'] = df['Pesan'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
          # URL and number features
        features['url_count'] = df['Pesan'].str.count(r'https?://|www\.')
        features['number_count'] = df['Pesan'].str.count(r'\d+')
        features['email_count'] = df['Pesan'].str.count(r'\S+@\S+')
        
        # Advanced linguistic features
        features['avg_word_length'] = df['clean_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x and x.split() else 0
        )
        
        print("✅ Fitur lanjutan berhasil diekstrak")
        return features
    
    def balance_dataset(self, df):
        """Balance dataset dengan teknik sampling yang aman dan robust"""
        print("⚖️ Melakukan balancing dataset...")
        
        ham_data = df[df['Kategori'] == 'ham'].copy()
        spam_data = df[df['Kategori'] == 'spam'].copy()
        
        print(f"Original - Ham: {len(ham_data)}, Spam: {len(spam_data)}")
        
        # Safety check untuk data kosong
        if len(ham_data) == 0 or len(spam_data) == 0:
            print("❌ Error: Salah satu class tidak memiliki data!")
            return df
        
        # Strategi balancing yang aman
        min_samples = min(len(ham_data), len(spam_data))
        max_samples = max(len(ham_data), len(spam_data))
        
        print(f"📊 Min samples: {min_samples}, Max samples: {max_samples}")
        
        # Jika perbedaan tidak terlalu besar (rasio < 3:1), gunakan undersample
        if max_samples / min_samples <= 3.0:
            target_size = min_samples
            print(f"🎯 Menggunakan undersample dengan target: {target_size}")
            
            ham_sample = ham_data.sample(n=min(target_size, len(ham_data)), random_state=42)
            spam_sample = spam_data.sample(n=min(target_size, len(spam_data)), random_state=42)
            
        else:
            # Jika perbedaan besar, gunakan strategi mixed
            target_size = min(max_samples, min_samples * 2)  # Maksimal 2x dari yang terkecil
            print(f"🎯 Menggunakan mixed strategy dengan target: {target_size}")
            
            if len(ham_data) < len(spam_data):
                # Ham lebih sedikit
                ham_sample = ham_data  # Ambil semua ham
                # Augmentasi ham jika perlu
                if len(ham_data) * 2 < len(spam_data):
                    print("🔄 Melakukan augmentasi ham...")
                    ham_augmented = self.augment_ham_data(ham_data)
                    ham_sample = pd.concat([ham_sample, ham_augmented]).head(target_size)
                
                # Sample spam sesuai target
                spam_sample = spam_data.sample(n=min(target_size, len(spam_data)), random_state=42)
            else:
                # Spam lebih sedikit
                spam_sample = spam_data  # Ambil semua spam
                # Augmentasi spam jika perlu
                if len(spam_data) * 2 < len(ham_data):
                    print("🔄 Melakukan augmentasi spam...")
                    spam_augmented = self.augment_spam_data(spam_data, target_ratio=2.0)
                    spam_sample = pd.concat([spam_sample, spam_augmented]).head(target_size)
                
                # Sample ham sesuai target
                ham_sample = ham_data.sample(n=min(target_size, len(ham_data)), random_state=42)
        
        # Gabungkan dan shuffle
        balanced_df = pd.concat([ham_sample, spam_sample], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Balanced - Ham: {len(balanced_df[balanced_df['Kategori'] == 'ham'])}, "
              f"Spam: {len(balanced_df[balanced_df['Kategori'] == 'spam'])}")
        
        return balanced_df
    
    def augment_ham_data(self, ham_data):
        """Augmentasi data ham (mirip dengan spam)"""
        print("🔄 Melakukan augmentasi data ham...")
        
        augmented_data = []
        
        for idx, row in ham_data.iterrows():
            # Variation: menambah variasi sederhana
            text = row['Pesan']
            varied_text = text + " Terima kasih." if not text.endswith('.') else text
            
            varied_row = row.copy()
            varied_row['Pesan'] = varied_text
            augmented_data.append(varied_row)
        
        return pd.DataFrame(augmented_data)
    
    def augment_spam_data(self, spam_data, target_ratio=1.0):
        """Augmentasi data spam dengan variasi teks"""
        print("🔄 Melakukan augmentasi data spam...")
        
        augmented_data = []
        target_count = int(len(spam_data) * target_ratio)
        
        # Simple augmentation techniques
        for idx, row in spam_data.iterrows():
            augmented_data.append(row)
            
            # Add variations if needed
            if len(augmented_data) < target_count:
                # Variation 1: Add some noise
                text = row['Pesan']
                # Simple text variation (could be more sophisticated)
                varied_text = text.replace('!', '!!') if '!' in text else text + '!'
                
                varied_row = row.copy()
                varied_row['Pesan'] = varied_text
                augmented_data.append(varied_row)
        
        return pd.DataFrame(augmented_data[:target_count])
    
    def preprocess_super_enhanced(self, df):
        """Preprocessing super enhanced"""
        print("\n🧹 Memulai preprocessing super enhanced...")
        
        df_processed = df.copy()
        
        # Advanced text cleaning
        print("📝 Membersihkan teks dengan metode super canggih...")
        df_processed['clean_text'] = df_processed['Pesan'].apply(self.super_advanced_text_cleaning)
        
        # Label encoding
        print("🏷️ Encoding label...")
        df_processed['label_encoded'] = df_processed['Kategori'].map({'ham': 0, 'spam': 1})
        
        # Remove empty texts
        df_processed = df_processed[df_processed['clean_text'].str.len() >= 3]
        
        # Balance dataset
        df_processed = self.balance_dataset(df_processed)
        
        # Extract advanced features
        features = self.extract_advanced_features(df_processed)
        df_processed = pd.concat([df_processed, features], axis=1)
        
        print(f"✅ Preprocessing selesai. Data final: {len(df_processed)} email")
        print(f"📊 Distribusi final:")
        print(df_processed['label_encoded'].value_counts())
        
        return df_processed
    
    def create_hybrid_features(self, texts):
        """Buat fitur hybrid: neural + traditional"""
        print("\n🔗 Membuat fitur hybrid...")
        
        # Neural features (sequences)
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(
                num_words=self.max_words,
                oov_token="<OOV>",
                filters='',  # Don't filter anything, we already cleaned
                lower=False,  # Already lowercased
                char_level=False
            )
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        neural_features = pad_sequences(
            sequences, 
            maxlen=self.max_length, 
            padding='post', 
            truncating='post'
        )
        
        # TF-IDF features
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        print(f"✅ Neural features shape: {neural_features.shape}")
        print(f"✅ TF-IDF features shape: {tfidf_features.shape}")
        
        return neural_features, tfidf_features
    
    def build_hybrid_model(self, tfidf_input_dim):
        """Build hybrid model dengan neural dan traditional features"""
        print("\n🏗️ Membangun hybrid model...")
        
        # Neural branch - untuk sequential patterns
        text_input = Input(shape=(self.max_length,), name='text_input')
        
        embedding = Embedding(
            input_dim=self.max_words,
            output_dim=300,  # Larger embedding
            input_length=self.max_length,
            trainable=True,
            name='embedding'
        )(text_input)
        
        # Bidirectional GRU layers
        gru1 = Bidirectional(GRU(
            128, return_sequences=True, 
            dropout=0.3, recurrent_dropout=0.3
        ), name='bi_gru_1')(embedding)
        
        gru2 = Bidirectional(GRU(
            64, return_sequences=True,
            dropout=0.3, recurrent_dropout=0.3
        ), name='bi_gru_2')(gru1)
        
        # Pooling layers
        max_pool = GlobalMaxPooling1D(name='global_max_pool')(gru2)
        avg_pool = GlobalAveragePooling1D(name='global_avg_pool')(gru2)
        
        # Combine pooling
        neural_features = Concatenate(name='concat_pools')([max_pool, avg_pool])
        neural_features = BatchNormalization(name='bn_neural')(neural_features)
        neural_features = Dropout(0.5, name='dropout_neural')(neural_features)
        
        # Traditional ML branch - untuk statistical features
        tfidf_input = Input(shape=(tfidf_input_dim,), name='tfidf_input')
        tfidf_dense = Dense(128, activation='relu', name='tfidf_dense_1')(tfidf_input)
        tfidf_dense = BatchNormalization(name='bn_tfidf')(tfidf_dense)
        tfidf_dense = Dropout(0.3, name='dropout_tfidf')(tfidf_dense)
        tfidf_features = Dense(64, activation='relu', name='tfidf_dense_2')(tfidf_dense)
        
        # Combine both branches
        combined = Concatenate(name='combine_features')([neural_features, tfidf_features])
        
        # Final layers
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='final_dense_1')(combined)
        x = BatchNormalization(name='bn_final_1')(x)
        x = Dropout(0.5, name='dropout_final_1')(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='final_dense_2')(x)
        x = BatchNormalization(name='bn_final_2')(x)
        x = Dropout(0.4, name='dropout_final_2')(x)
        
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001), name='final_dense_3')(x)
        x = Dropout(0.3, name='dropout_final_3')(x)
        
        # Output
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(inputs=[text_input, tfidf_input], outputs=output)
          # Compile with focal loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Hybrid model berhasil dibangun!")
        print("\n📋 Arsitektur Model:")
        model.summary()
        
        self.model = model
        return model
    
    def lr_schedule(self, epoch, lr):
        """Learning rate scheduler"""
        if epoch < 10:
            return float(lr)
        else:
            return float(lr * np.exp(-0.1))
    
    def train_hybrid_model(self, X_neural, X_tfidf, y, epochs=30, batch_size=32):
        """Train hybrid model dengan advanced techniques"""
        print(f"\n🚀 Training hybrid model untuk {epochs} epochs...")
        
        # Split data
        indices = np.arange(len(y))
        X_neural_train, X_neural_val, X_tfidf_train, X_tfidf_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_neural, X_tfidf, y, indices, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📈 Training: {len(y_train)}, Validation: {len(y_val)}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            LearningRateScheduler(self.lr_schedule, verbose=0)
        ]
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"📊 Class weights: {class_weight_dict}")
        
        # Training
        history = self.model.fit(
            [X_neural_train, X_tfidf_train], y_train,
            validation_data=([X_neural_val, X_tfidf_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        print("✅ Training selesai!")
        return history, (X_neural_val, X_tfidf_val, y_val)
    
    def evaluate_hybrid_model(self, X_neural_test, X_tfidf_test, y_test):
        """Evaluasi comprehensive hybrid model"""
        print("\n📊 Evaluasi comprehensive hybrid model...")
        
        # Predict
        y_pred_proba = self.model.predict([X_neural_test, X_tfidf_test])
        
        # Find optimal threshold
        best_threshold = 0.5
        best_f1 = 0
        
        print("🔍 Mencari threshold optimal:")
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_pred_proba > threshold).astype(int).flatten()
            f1 = f1_score(y_test, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"🏆 Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
        
        # Final evaluation
        y_pred_optimal = (y_pred_proba > best_threshold).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred_optimal)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"🎯 Final Accuracy: {accuracy:.4f}")
        print(f"📈 AUC Score: {auc_score:.4f}")
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_optimal, target_names=['Ham', 'Spam']))
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'best_threshold': best_threshold,
            'y_pred': y_pred_optimal,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred_optimal)
        }
    
    def predict_email_hybrid(self, email_text, threshold=0.5):
        """Prediksi email dengan hybrid model"""
        # Clean text
        clean_text = self.super_advanced_text_cleaning(email_text)
        
        if not clean_text:
            return {
                'prediction': 'HAM',
                'probability': 0.0,
                'confidence': 0.5,
                'note': 'Teks kosong setelah cleaning'
            }
        
        # Neural features
        sequence = self.tokenizer.texts_to_sequences([clean_text])
        neural_input = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # TF-IDF features
        tfidf_input = self.tfidf_vectorizer.transform([clean_text]).toarray()
        
        # Predict
        probability = self.model.predict([neural_input, tfidf_input], verbose=0)[0][0]
        prediction = "SPAM" if probability > threshold else "HAM"
        confidence = max(probability, 1-probability)
        
        return {
            'prediction': prediction,
            'probability': float(probability),
            'confidence': float(confidence),
            'clean_text': clean_text,
            'threshold_used': threshold
        }

def main():
    """Main function dengan semua improvements"""
    print("🚀 SUPER ENHANCED EMAIL SPAM CLASSIFIER WITH HYBRID MODEL")
    print("=" * 80)
    
    # Initialize
    classifier = SuperEnhancedEmailClassifier(max_words=15000, max_length=200)
    
    # 1. Load and analyze data
    df = classifier.load_and_analyze_data()
    if df is None:
        return
    
    # 2. Super enhanced preprocessing
    df_processed = classifier.preprocess_super_enhanced(df)
    
    # 3. Create hybrid features
    neural_features, tfidf_features = classifier.create_hybrid_features(
        df_processed['clean_text'].tolist()
    )
    y = df_processed['label_encoded'].values
    
    # 4. Build hybrid model
    model = classifier.build_hybrid_model(tfidf_features.shape[1])
    
    # 5. Train hybrid model
    history, (X_neural_val, X_tfidf_val, y_val) = classifier.train_hybrid_model(
        neural_features, tfidf_features, y, epochs=30, batch_size=32
    )
    
    # 6. Evaluate model
    results = classifier.evaluate_hybrid_model(X_neural_val, X_tfidf_val, y_val)
    
    # 7. Test with real examples
    optimal_threshold = results['best_threshold']
    
    print(f"\n🧪 Testing dengan threshold optimal: {optimal_threshold:.2f}")
    print("-" * 60)
    
    # Spam examples
    spam_examples = [
        "SELAMAT!!! Anda MENANG hadiah 100 JUTA! Klik link: http://scam.com GRATIS!",
        "Investasi mudah untung 500%! Modal 100rb jadi 500rb! WA: 08123456789",
        "PROMO GILA!!! Beli 1 GRATIS 10! Harga murah banget! BURUAN ORDER!",
        "Dapatkan uang MUDAH tanpa modal! Gabung sekarang dapat BONUS 50%!"
    ]
    
    # Ham examples  
    ham_examples = [
        "Halo pak, mohon konfirmasi untuk rapat hari Senin jam 10 pagi.",
        "Laporan bulanan sudah selesai, mohon review dan feedback.",
        "Terima kasih atas presentasinya kemarin, sangat informatif.",
        "Meeting project review dijadwalkan Rabu pukul 14.00 di ruang konferensi."
    ]
    
    print("📧 TESTING SPAM EMAILS:")
    for i, email in enumerate(spam_examples, 1):
        result = classifier.predict_email_hybrid(email, threshold=optimal_threshold)
        print(f"\n{i}. {email[:60]}...")
        print(f"   🎯 Prediksi: {result['prediction']}")
        print(f"   📊 Probabilitas: {result['probability']:.4f}")
        print(f"   💯 Confidence: {result['confidence']:.4f}")
    
    print("\n📧 TESTING HAM EMAILS:")
    for i, email in enumerate(ham_examples, 1):
        result = classifier.predict_email_hybrid(email, threshold=optimal_threshold)
        print(f"\n{i}. {email[:60]}...")
        print(f"   🎯 Prediksi: {result['prediction']}")
        print(f"   📊 Probabilitas: {result['probability']:.4f}")
        print(f"   💯 Confidence: {result['confidence']:.4f}")
    
    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"🎯 Final Accuracy: {results['accuracy']:.4f}")
    print(f"📈 AUC Score: {results['auc_score']:.4f}")
    print(f"⚖️ Optimal Threshold: {optimal_threshold:.2f}")

if __name__ == "__main__":
    main()