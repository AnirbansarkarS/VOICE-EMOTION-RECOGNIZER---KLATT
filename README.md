<h1 align="center">🎙️ Voice Emotion Recognition</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-brightgreen?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/github/languages/top/anirbanSarkars/voice-emotion-recognizer?style=flat-square" />
  <img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F%20by%20Anirban-blueviolet?style=flat-square" />
</p>

---

## 🧠 What It Does

**Voice Emotion Recognition** is a real-time web application that listens to your voice and **detects your emotion** in seconds — powered by machine learning and a minimalistic Streamlit UI.

Whether you're feeling 🎉 happy, 😠 angry, 😢 sad, or 😐 neutral — the app understands you.

---

## 🌟 Highlights

🚀 Live voice recording  
🎧 Upload `.wav` files for prediction  
🤖 ML-powered emotion classification  
📊 Visual probability pie chart  
🎨 Simple, clean UI with one-click predictions

---

## 📽 Demo Preview

> 🎬 _Coming Soon_: You can host this on [Streamlit Cloud](https://streamlit.io/cloud) for others to try.

---

## 🛠 Built With

| Technology       | Role                            |
|------------------|----------------------------------|
| `Python 3.x`     | Core Language                   |
| `Streamlit`      | Web App Framework               |
| `scikit-learn`   | ML Model Training & Inference   |
| `librosa`        | Audio Feature Extraction        |
| `sounddevice`    | Live Audio Recording            |
| `matplotlib`     | Visualization (Pie Chart)       |

---

## 🗂 Project Structure

voice-emotion-recognizer/
│
├── app.py # Main Streamlit App
├── models/
│ └── emotion_model.pkl # Pre-trained Emotion Classifier
├── utils/
│ └── audio_features.py # Feature Extraction Code
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🧪 Try It Locally

```bash
git clone https://github.com/anirbanSarkars/voice-emotion-recognizer.git
cd voice-emotion-recognizer

# Create & activate virtual environment
python -m venv emotion_track
emotion_track\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
Then open http://localhost:8501 in your browser.

📦 Dependencies
txt
Copy
Edit
streamlit
numpy<=2.2
sounddevice
scikit-learn
librosa
matplotlib
joblib
soundfile
✅ Note: numpy<=2.2 is required to ensure compatibility with numba.

💡 How It Works
You record or upload an audio file.

The app extracts MFCC features using librosa.

A trained RandomForestClassifier predicts the emotion.

The app visualizes probabilities via a pie chart.

📚 Dataset
Model trained on the RAVDESS Speech Emotion Dataset, containing emotional speech recordings across multiple speakers.

🔗 Live Hosting Options
Platform	Supports Streamlit
Streamlit Cloud	✅
Render	✅
GitHub Pages	❌ (only static sites)

🙌 Acknowledgements
This project was built with passion by Anirban Sarkar — a curious learner, coder, and innovator dedicated to building intelligent systems that understand human behavior.

🤝 Contribute
Want to add more emotion categories, retrain the model, or make the UI even cooler?
Fork this repo, build your magic, and make a PR!

📜 License
MIT License.
Feel free to use and modify with credit.

🚀 Final Thoughts
"Emotion is energy in motion. Let's help machines understand it too."

If you liked this project — drop a ⭐ on GitHub and share it!
Let’s make emotion recognition more accessible, one voice at a time.
