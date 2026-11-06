# Book Genre Classifier 📚

A machine learning web application that classifies book descriptions into 12 genres using TF-IDF feature extraction and Random Forest ensemble modeling.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🎯 Model Performance & Architecture

- **Test Accuracy**: 83.3% - 91.7% on realistic book scenarios
- **Genres Supported (12)**: Mystery, Romance, Fantasy, Science Fiction, Horror, Thriller, Young Adult, Biography, Non-Fiction, Literary Fiction, Self-Help, Historical Fiction
- **Dataset**: 480 balanced, keyword-focused descriptions across all 12 classes
- **Feature Extraction**: TF-IDF Vectorizer with unigrams, bigrams, and trigrams (8,000 max features)
- **Classifier**: Random Forest Classifier (200 estimators, max depth 25)
- **Deployment Interface**: Streamlit with interactive Plotly metrics and confidence scoring

---

## 🌐 Public Cloud Deployment (Streamlit Community Cloud)

You can deploy this application directly to **Streamlit Community Cloud** for free with a public live URL.

### Deployment Steps:

1. **Sign in to Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.

2. **Create New App**:
   - Click the **"New app"** button.
   - Select your repository: `meignanavignesh/bookgenre-classifier`
   - Set **Branch**: `main`
   - Set **Main file path**: `app.py`
   - (Optional) In **Advanced settings**, ensure Python version is set to `3.10`, `3.11`, or `3.12`.

3. **Deploy**:
   - Click **"Deploy!"**
   - Streamlit Cloud will automatically clone the repo, install dependencies from `requirements.txt`, load the pre-trained model from `models/final_model.pkl`, and launch the web UI.

4. **Live URL**:
   - Your public demo will be accessible worldwide at:
     `https://<your-custom-subdomain>.streamlit.app`

---

## 🚀 Local Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/meignanavignesh/bookgenre-classifier.git
cd bookgenre-classifier
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

Or use the unified CLI manager:
```bash
python main.py web
```

---

## 💻 Command-Line Interface (CLI)

### Predict Genre via CLI:
```bash
# Using predict_final.py directly:
python predict_final.py "A detective investigates a series of murders in Victorian London"

# Or using main.py:
python main.py predict "A young wizard discovers magical powers at an enchanted academy"
```

### Train / Retrain Model:
```bash
python main.py train
# Or directly:
python final_improved_model.py
```

### Run Automated Tests:
```bash
python -m pytest -v
```

---

## 📁 Project Structure

```
bookgenre-classifier/
├── .streamlit/
│   └── config.toml           # Streamlit server & theme configuration
├── data/
│   └── final_books.csv       # Balanced 12-genre dataset (480 books)
├── models/
│   └── final_model.pkl       # Serialized Random Forest + TF-IDF pipeline
├── tests/
│   ├── test_app.py           # Streamlit UI & inference integration tests
│   └── test_model.py         # Model loading, classes, and probability tests
├── app.py                    # Streamlit web application entry point
├── config.py                 # Project path and hyperparameter configuration
├── final_improved_model.py   # Dataset generation, pipeline training, and evaluation
├── main.py                   # Central CLI controller (predict, web, train)
├── predict_final.py          # Standalone CLI prediction script
├── requirements.txt          # Streamlit Cloud production dependencies
├── requirements-dev.txt      # Testing and development dependencies
├── PROJECT_SUMMARY.md        # Academic summary and evaluation report
└── README.md                 # Project documentation and deployment guide
```

---

## 🎯 Supported Genres

| # | Genre | Description Focus |
|---|---|---|
| 1 | **Mystery** | Detective stories, locked-room puzzles, forensic investigations |
| 2 | **Romance** | Love stories, relationships, romantic journeys |
| 3 | **Fantasy** | Magic, wizards, dragons, mythical kingdoms |
| 4 | **Science Fiction** | Space exploration, AI, futuristic technology, alien encounters |
| 5 | **Horror** | Supernatural terror, haunted settings, psychological dread |
| 6 | **Thriller** | Espionage, conspiracies, fast-paced chases, suspense |
| 7 | **Young Adult** | Teenage coming-of-age, high school life, identity |
| 8 | **Biography** | Life chronicles of historical and influential figures |
| 9 | **Non-Fiction** | Science, philosophy, academic research, historical accounts |
| 10 | **Fiction** | Contemporary stories, literary fiction, human experiences |
| 11 | **Self-Help** | Personal development, productivity, wellness, habit formation |
| 12 | **Historical Fiction** | Narratives set against historical eras and events |

---

## 🛡️ Security & Resilience

- **No Secrets or Credentials Exposed**: Clean codebase with zero hardcoded API keys or sensitive credentials.
- **Self-Healing Fallback**: If `models/final_model.pkl` is ever deleted or missing on a new environment, `app.py` automatically generates the training dataset and trains the model in seconds during initialization.
- **Path-Independent Execution**: All asset references use absolute resolution based on `Path(__file__)`, ensuring zero `FileNotFoundError` regardless of where the app is launched from.
