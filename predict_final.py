"""
Use the final improved model for accurate predictions
"""

import pickle
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"

def predict_genre(description, model_path=MODEL_PATH):
    """Predict genre using the final improved model"""
    try:
        # Load the final model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction
        prediction = model.predict([description])[0]
        probabilities = model.predict_proba([description])[0]
        confidence = float(max(probabilities))
        
        return prediction, confidence
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_final.py 'Your book description here'")
        return
    
    description = sys.argv[1]
    
    print("🎯 FINAL MODEL PREDICTION:")
    print("=" * 60)
    print(f"Description: {description}")
    print("-" * 60)
    
    genre, confidence = predict_genre(description)
    
    if genre:
        print(f"Predicted Genre: {genre.replace('_', ' ').title()}")
        print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        
        if confidence >= 0.6:
            print("🎯 High confidence prediction!")
        elif confidence >= 0.4:
            print("⚠️ Moderate confidence")
        else:
            print("❓ Low confidence - description might be ambiguous")
    else:
        print("❌ Failed to make prediction")

if __name__ == "__main__":
    main()