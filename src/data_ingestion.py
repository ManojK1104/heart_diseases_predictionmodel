import numpy as np
import pandas as pd

def ingestion():
    df=pd.read_csv(r"https://raw.githubusercontent.com/ManojK1104/heart_diseases_predictionmodel/refs/heads/main/data/heart_disease_risk_2026.csv")
    return df