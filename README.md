# SMS-SPAM-DETECTION-LSTM
# 📩 SMS Spam Detection using LSTM

This project classifies SMS messages as **Spam** or **Ham (Not Spam)** using a **Long Short-Term Memory (LSTM)** neural network.

---

## 🧠 Overview
This model uses deep learning and NLP preprocessing techniques to analyze SMS text messages.  
It automatically downloads the dataset from Kaggle (UCI SMS Spam Collection) and trains an LSTM model to detect spam messages.

---

## ⚙️ Steps to Run
 1️⃣ Clone the Repository
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

📦 Requirements

Install all dependencies using requirements.txt.
👨‍💻 Author

Name: ENAPA JEEVA SRI
Project: SMS Spam Detection using LSTM
Tools: Python, TensorFlow, KaggleHub, VS Code



