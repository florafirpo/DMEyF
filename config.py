"""
Configuración del proyecto - Todo en un solo lugar

CÓMO CAMBIAR PARÁMETROS:
------------------------

1. CAMBIAR SEMILLAS (para más/menos modelos o diferentes resultados):
   SEMILLAS = np.array([123, 456, 789])  # Solo 3 modelos (más rápido)
   SEMILLAS = np.array([111, 222, 333, 444, 555, 666])  # 6 modelos (más lento)

2. CAMBIAR HIPERPARÁMETROS DE LIGHTGBM (se cargan del JSON, pero puedes override):
   En el script, después de cargar config, agrega:
   config['params']['num_leaves'] = 150  # Más hojas = más complejo
   config['params']['learning_rate'] = 0.03  # Más bajo = más iteraciones necesita

4. CAMBIAR ARCHIVOS DE DATOS:
   RUTAS['train_123'] = 'data/mi_archivo_train.csv'
   
5. CAMBIAR PESOS DE LAS CLASES (para dar más/menos importancia):
   PESOS['BAJA+2'] = 1.00003  # Más peso a BAJA+2
   PESOS['BAJA+1'] = 1.000005 # Menos peso a BAJA+1
"""
import numpy as np
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Semillas para reproducibilidad
SEMILLAS = np.array([550007, 550019, 550031, 550033, 550047])

# Rutas de archivos
RUTAS = {
    'train_123': 'data/df_train_01_02_03.csv',
    'train_c12': 'data/df_train_c12.csv',
    'test_04': 'data/df_test_04.csv',
    'kaggle': 'data/df_kaggle.csv',
    'hiperparametros': 'data/mejores_hiperparametros.json'
}

# Parámetros de negocio
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000

# Pesos por clase
PESOS = {
    'BAJA+2': 1.00002,
    'BAJA+1': 1.00001,
    'CONTINUA': 1.0
}