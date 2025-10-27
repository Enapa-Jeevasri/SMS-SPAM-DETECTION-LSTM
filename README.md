# SMS-SPAM-DETECTION-LSTM
# 📩 SMS Spam Detection using LSTM

This project classifies SMS messages as **Spam** or **Ham (Not Spam)** using a **Long Short-Term Memory (LSTM)** neural network.

---

## 🧠 Overview
This model uses deep learning and NLP preprocessing techniques to analyze SMS text messages.  
It automatically downloads the dataset from Kaggle (UCI SMS Spam Collection) and trains an LSTM model to detect spam messages.

---

## ⚙️ Steps to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/sms-spam-detection-lstm.git
cd sms-spam-detection-lstm
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # For Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Model
python sms_spam_lstm.py
🧱 Model Architecture

Embedding Layer (64 units)

LSTM Layer (64 units, Dropout + Recurrent Dropout)

Dense Layer (32 units, ReLU)

Output Layer (Sigmoid Activation)

Loss: Binary Crossentropy
Optimizer: Adam
Metric: Accuracy
🧪 Sample Output
✅ Dataset downloaded at: C:\Users\Admin\.cache\kagglehub\datasets\uciml\sms-spam-collection-dataset\versions\1
🚀 Training model...
Epoch 1/5
...
✅ Test Accuracy: 0.9821

Message: Congratulations! You have won a free iPhone!
Predicted: Spam

Message: Hey, are we meeting for lunch today?
Predicted: Ham
📦 Requirements

Install all dependencies using requirements.txt.
👨‍💻 Author

Name: ENAPA JEEVA SRI
Project: SMS Spam Detection using LSTM
Tools: Python, TensorFlow, KaggleHub, VS Code

---

## 📦 requirements.txt

```txt
tensorflow==2.17.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
kagglehub==0.2.6

