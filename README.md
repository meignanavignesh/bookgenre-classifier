# Book Genre Classifier

A machine learning project that accurately classifies books into 12 genres based on their descriptions using advanced NLP techniques.

## 🎯 Final Model Performance
- **Accuracy**: 83.3% on test cases
- **Genres Supported**: 12 (Mystery, Romance, Fantasy, Sci-Fi, Horror, Thriller, YA, Biography, Non-Fiction, Fiction, Self-Help, Historical Fiction)
- **Dataset**: 480 high-quality, balanced descriptions
- **Technology**: Random Forest + TF-IDF + N-grams

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model (Optional - model already trained)
```bash
python final_improved_model.py
```

### 3. Predict Genre
```bash
python predict_final.py "Your book description here"
```

### 4. Web Interface
```bash
python -m streamlit run app.py
```
Then open: http://localhost:8501

## 📁 Project Structure

```
book-genre-classifier/
├── data/
│   └── final_books.csv           # Training dataset (480 entries)
├── models/
│   └── final_model.pkl           # Trained model (83.3% accuracy)
├── results/                      # Evaluation results
├── notebooks/                    # Data exploration
├── app.py                        # Streamlit web interface
├── predict_final.py              # Command-line prediction
├── final_improved_model.py       # Model training script
├── config.py                     # Configuration
└── requirements.txt              # Dependencies
```

## 🎯 Supported Genres

1. **Mystery** - Detective stories, crime investigations
2. **Romance** - Love stories, relationships
3. **Fantasy** - Magic, dragons, mythical creatures
4. **Science Fiction** - Future, space, technology
5. **Horror** - Scary, supernatural, terror
6. **Thriller** - Action, suspense, chase scenes
7. **Young Adult** - Teenage protagonists, coming-of-age
8. **Biography** - Life stories of real people
9. **Non-Fiction** - Educational, factual content
10. **Fiction** - Literary fiction, contemporary stories
11. **Self-Help** - Personal development, guides
12. **Historical Fiction** - Stories set in historical periods

## 💡 Usage Examples

```bash
# Mystery
python predict_final.py "Detective investigates murders in Victorian London"

# Romance  
python predict_final.py "Two people fall in love despite family opposition"

# Fantasy
python predict_final.py "Young wizard discovers magical powers at academy"

# Science Fiction
python predict_final.py "Space colonists fight aliens on distant planet"
```

## 🎓 Academic Features

- **Balanced Dataset**: Equal representation across all genres
- **Advanced NLP**: TF-IDF vectorization with n-grams
- **Multiple Models**: Comparison of different algorithms
- **Error Analysis**: Comprehensive evaluation metrics
- **Real-world Testing**: Works with user-provided descriptions