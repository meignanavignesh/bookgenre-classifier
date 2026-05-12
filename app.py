"""
Streamlit Web UI for Book Genre Classifier
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.model import GenreClassifier
import json

# Page configuration
st.set_page_config(
    page_title="Book Genre Classifier",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .genre-result {
        font-size: 2rem;
        font-weight: bold;
        color: #2e8b57;
        text-align: center;
        padding: 1rem;
        border: 2px solid #2e8b57;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 1.5rem;
        color: #ff6347;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    try:
        classifier = GenreClassifier()
        classifier.load_model('naive_bayes')
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_dataset_info():
    """Load dataset information"""
    try:
        df = pd.read_csv('data/raw_books.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Book Genre Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Predict the genre of any book based on its description!")
    
    # Load model
    classifier = load_model()
    if not classifier:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # Sidebar with project info
    with st.sidebar:
        st.header("📊 Project Information")
        
        # Dataset info
        df = load_dataset_info()
        if df is not None:
            st.metric("Total Books", f"{len(df):,}")
            st.metric("Genres Supported", df['genre'].nunique())
            
            # Genre distribution chart
            st.subheader("Genre Distribution")
            genre_counts = df['genre'].value_counts()
            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title="Books per Genre"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Supported genres
        st.subheader("🎯 Supported Genres")
        genres = [
            "Biography", "Fantasy", "Fiction", "Historical Fiction",
            "Horror", "Mystery", "Non-Fiction", "Romance", 
            "Science Fiction", "Self-Help", "Thriller", "Young Adult"
        ]
        for i, genre in enumerate(genres, 1):
            st.write(f"{i:2d}. {genre}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Book Description")
        
        # Text input
        description = st.text_area(
            "Book Description:",
            placeholder="Enter a book description here... (e.g., 'A detective investigates a murder in Victorian London')",
            height=150,
            help="Enter a detailed description of the book. The more specific details you provide, the better the prediction will be."
        )
        
        # Example descriptions
        st.subheader("💡 Try These Examples:")
        examples = {
            "Mystery": "A detective investigates a puzzling murder in Victorian London with unexpected twists.",
            "Fantasy": "A young wizard discovers magical powers and must save the kingdom from dark forces.",
            "Romance": "Two hearts find love against all odds in this touching story of redemption.",
            "Horror": "Something sinister lurks in the shadows of the haunted mansion, terrorizing visitors.",
            "Sci-Fi": "In the year 3000, robots control the planet and humans must fight for survival.",
            "Biography": "The inspiring life story of Einstein and his journey to scientific greatness."
        }
        
        example_cols = st.columns(3)
        for i, (genre, example) in enumerate(examples.items()):
            col_idx = i % 3
            with example_cols[col_idx]:
                if st.button(f"{genre}", key=f"example_{i}"):
                    st.session_state.example_text = example
        
        # Use example text if button was clicked
        if 'example_text' in st.session_state:
            description = st.session_state.example_text
            st.text_area("Selected Example:", value=description, height=100, disabled=True)
    
    with col2:
        st.subheader("🎯 Prediction Result")
        
        if st.button("🔮 Predict Genre", type="primary", use_container_width=True):
            if description.strip():
                with st.spinner("Analyzing description..."):
                    try:
                        # Make prediction
                        genre, confidence = classifier.predict_genre(description)
                        
                        # Display result
                        st.markdown(f'<div class="genre-result">{genre.replace("_", " ").title()}</div>', 
                                  unsafe_allow_html=True)
                        st.markdown(f'<div class="confidence-score">Confidence: {confidence:.1%}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.progress(confidence)
                        
                        # Interpretation
                        if confidence >= 0.8:
                            st.success("🎯 High confidence prediction!")
                        elif confidence >= 0.6:
                            st.warning("⚠️ Moderate confidence - consider adding more details")
                        else:
                            st.error("❓ Low confidence - description might be ambiguous")
                            
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
            else:
                st.warning("Please enter a book description first!")
    
    # Additional information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🚀 How It Works")
        st.write("""
        1. **Text Processing**: Cleans and preprocesses the description
        2. **Feature Extraction**: Uses TF-IDF, sentiment analysis, and statistical features
        3. **ML Prediction**: Gaussian Naive Bayes model classifies the genre
        4. **Confidence Score**: Shows how certain the model is about the prediction
        """)
    
    with col2:
        st.subheader("📈 Model Performance")
        st.metric("Training Accuracy", "96.1%")
        st.metric("Test Accuracy", "96.1%")
        st.metric("Feature Dimensions", "5,731")
        st.metric("Training Examples", "2,168")
    
    with col3:
        st.subheader("💡 Tips for Better Results")
        st.write("""
        - Include **specific keywords** (detective, wizard, romance)
        - Mention **setting** and **characters**
        - Describe **plot elements** and **themes**
        - Use **2-3 sentences** for best results
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**📚 Book Genre Classifier** - Academic ML Project | Built with Streamlit")

if __name__ == "__main__":
    main()