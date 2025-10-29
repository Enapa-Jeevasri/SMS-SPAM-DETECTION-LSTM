# SMS Spam Detection using LSTM
# Author: Your Name
# Description: Classifies SMS messages as Spam or Ham using LSTM

import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# ----------------------------------------------------------
# STEP 1: Download dataset from Kaggle
# ----------------------------------------------------------
print("📥 Downloading dataset...")
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("✅ Dataset downloaded at:", path)

# Load dataset file
data_path = f"{path}/spam.csv"  # KaggleHub saves as spam.csv
df = pd.read_csv(data_path, encoding='latin-1')

# ----------------------------------------------------------
# STEP 2: Clean and prepare data
# ----------------------------------------------------------
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Sample data:")
print(df.head())

# Encode labels: ham = 0, spam = 1
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# ----------------------------------------------------------
# STEP 3: Split train/test
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# STEP 4: Tokenize text
# ----------------------------------------------------------
max_words = 5000   # vocabulary size
max_len = 100      # sequence length

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# ----------------------------------------------------------
# STEP 5: Build LSTM model
# ----------------------------------------------------------
model = Sequential([
    Embedding(max_words, 64, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ----------------------------------------------------------
# STEP 6: Train model
# ----------------------------------------------------------
print("🚀 Training model...")
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64,
                    validation_data=(X_test_pad, y_test))

# ----------------------------------------------------------
# STEP 7: Evaluate model
# ----------------------------------------------------------
loss, acc = model.evaluate(X_test_pad, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# ----------------------------------------------------------
# STEP 8: Test with custom input
# ----------------------------------------------------------
sample_sms = ["Congratulations! You have won a free iPhone!", 
              "Hey, are we meeting for lunch today?"]

sample_seq = tokenizer.texts_to_sequences(sample_sms)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post')
predictions = (model.predict(sample_pad) > 0.5).astype(int)

for i, msg in enumerate(sample_sms):
    label = "Spam" if predictions[i] == 1 else "Ham"
    print(f"Message: {msg}\nPredicted: {label}\n")