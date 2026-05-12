"""
Main script for Book Genre Classifier
Simple interface to access all functionality
"""

import sys
import subprocess
import os

def show_help():
    """Show usage instructions"""
    print("📚 BOOK GENRE CLASSIFIER")
    print("=" * 50)
    print("🎯 Accuracy: 83.3% | Genres: 12 | Dataset: 480 books")
    print("=" * 50)
    print()
    print("USAGE:")
    print("  python main.py predict 'Your book description here'")
    print("  python main.py web")
    print("  python main.py train")
    print("  python main.py help")
    print()
    print("EXAMPLES:")
    print("  python main.py predict 'Detective investigates murder in London'")
    print("  python main.py predict 'Young wizard discovers magical powers'")
    print("  python main.py web  # Launch web interface")
    print()
    print("SUPPORTED GENRES:")
    genres = [
        "Mystery", "Romance", "Fantasy", "Science Fiction", 
        "Horror", "Thriller", "Young Adult", "Biography",
        "Non-Fiction", "Fiction", "Self-Help", "Historical Fiction"
    ]
    for i, genre in enumerate(genres, 1):
        print(f"  {i:2d}. {genre}")

def predict_genre(description):
    """Predict genre using the final model"""
    try:
        result = subprocess.run([
            sys.executable, 'predict_final.py', description
        ], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Make sure the model is trained. Run: python main.py train")

def launch_web():
    """Launch the web interface"""
    try:
        print("🌐 Launching web interface...")
        print("📍 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped")
    except Exception as e:
        print(f"Error launching web interface: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")

def train_model():
    """Train the model"""
    try:
        print("🔄 Training model...")
        result = subprocess.run([
            sys.executable, 'final_improved_model.py'
        ], check=True)
        print("✅ Model training complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error training model: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'predict':
        if len(sys.argv) < 3:
            print("❌ Error: Please provide a book description")
            print("Usage: python main.py predict 'Your book description here'")
            return
        description = sys.argv[2]
        predict_genre(description)
        
    elif command == 'web':
        launch_web()
        
    elif command == 'train':
        train_model()
        
    elif command == 'help':
        show_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python main.py help' for usage instructions")

if __name__ == "__main__":
    main()