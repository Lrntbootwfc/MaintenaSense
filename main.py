"""
main.py
End-to-end runner for the Predictive Maintenance project.
"""

import os
from src import collect_data as data_collection
from src import clean_data as data_preprocessing
from src import train_model as predictive_model

from src import presentation as presentation_generator
DATA_DIR = "data"



RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "models"
REPORTS_DIR = "reports"
PRESENTATION_DIR = "presentation/assets"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PRESENTATION_DIR, exist_ok=True)

def run_all():
    print("1) Generating simulated raw data...")
    data_collection.generate_simulated_data(output_dir=RAW_DIR, n_machines=10, days=120)

    print("2) Preprocessing data...")
    data_preprocessing.run_preprocessing(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)

    print("3) Training model...")
    predictive_model.train_and_evaluate(processed_dir=PROCESSED_DIR, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR)
    
    print("4) Generating presentation (PPTX)...")
    presentation_generator.generate_presentation(reports_dir=REPORTS_DIR, presentation_dir=PRESENTATION_DIR)

    print("5) Dashboard ready — run `python -m src.dashboard` or `streamlit run src/dashboard.py` to view interactive UI.")

if __name__ == "__main__":
    run_all()
