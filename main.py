import polars as pl
from pathlib import Path
import os
import datetime
import logging
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier

# main.py - Script principal para preparar datos, entrenar modelos y generar predicciones
from config import CONFIG, SEMILLAS
from data_prep import preparar_datos_completos
from training import entrenar_ensemble
from prediction import generar_submission

# 1. Preparar datos (reutilizable)
X_full, y_full, pesos = preparar_datos_completos(
    train_path="df_train.csv",
    test_path="df_test.csv"
)

# 2. Entrenar modelos (una sola vez)
modelos = entrenar_ensemble(
    X_full, y_full, pesos,
    config=CONFIG,
    semillas=SEMILLAS
)

# 3. Generar predicciones
submission = generar_submission(
    modelos=modelos,
    kaggle_path="df_kaggle.csv",
    threshold=0.025
)

# 4. Guardar
submission.to_csv("submission.csv", index=False)