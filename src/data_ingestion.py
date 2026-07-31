import numpy as np
import pandas as pd

def ingestion():
    df=pd.read_csv(r"E:\heart_diseases_predictionmodel\data\heart_disease_risk_2026.csv")
    return df