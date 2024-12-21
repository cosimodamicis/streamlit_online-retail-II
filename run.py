"""
Script per avviare la Streamlit app
"""
import streamlit.web.bootstrap as bootstrap
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "src" / "app.py"
    bootstrap.run(str(app_path), '', [], [])

if __name__ == "__main__":
    main()