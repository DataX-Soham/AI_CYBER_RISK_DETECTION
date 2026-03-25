🔥 AI Cyber Risk Detection System

An advanced Machine Learning project that detects potential cyber threats and assigns a risk score based on network behavior.

---

🚀 Overview

This project combines Machine Learning and Cybersecurity concepts to analyze network activity and classify whether it is normal or malicious.

It uses:

- Regression → to calculate a risk score
- Classification → to detect attack vs normal behavior

---

🧠 Concepts Used

- Linear Regression (from scratch)
- Softmax / Logistic Regression
- Gradient Descent Optimization
- Cross Entropy Loss
- Data Preprocessing & Normalization
- Confusion Matrix Evaluation

---

📊 Dataset

- KDD Cup 99 Intrusion Detection Dataset
- Loaded using Scikit-learn

---

⚙️ Features

- Detects cyber attacks from network data
- Generates a numerical risk score
- Classifies traffic as Normal / Attack
- Provides simple explanations for predictions
- Visualizes:
  - Training loss curves
  - Confusion matrix
  - Risk distribution

---

📈 Sample Output

Risk Score: 78
Risk Level: ATTACK ⚠️

Reasons:

- High data transfer
- Unusual traffic pattern

---

📁 Project Structure

smart-risk-intelligence-system/
│
├── src/
│   ├── preprocessing.py
│   ├── regression.py
│   ├── classification.py
│   ├── evaluation.py
│   ├── explainability.py
│
├── output/
│   └── graphs/
│
├── main.py
├── README.md

---

▶️ How to Run

1. Install dependencies:
   pip install numpy matplotlib scikit-learn

2. Run the project:
   python main.py

---

📊 Output Graphs

The system automatically saves graphs in:
output/graphs/

- Regression Loss Curve
- Classification Loss Curve
- Confusion Matrix
- Risk Score Distribution

---

💡 Future Improvements

- Real-time network monitoring
- Integration with security tools
- Deployment as a web app
- Advanced anomaly detection

---

👨‍💻 Author

DataX_Soham

---

⭐ If you like this project

Give it a star on GitHub!