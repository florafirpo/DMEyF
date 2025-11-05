"""
Script 2: Predecir en Kaggle (mes 06)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

from config import SEMILLAS, RUTAS
from utils import preparar_para_modelo, predecir_ensemble, crear_submission


# ============================================================
# CONFIGURACIÓN
# ============================================================
NOMBRE_EXPERIMENTO = "exp_01_02_03"  # Debe coincidir con el del script 1
N_ENVIOS = 12000  # Número fijo de envíos o None para usar threshold
THRESHOLD = 0.025  # Umbral de probabilidad (si N_ENVIOS es None)


# ============================================================
# PASO 1: CARGAR MODELOS
# ============================================================
print("\n" + "="*60)
print("PASO 1: CARGAR MODELOS")
print("="*60)

modelos = []
for semilla in SEMILLAS:
    path = f'data/modelo_{NOMBRE_EXPERIMENTO}_semilla_{semilla}.txt'
    modelo = lgb.Booster(model_file=path)
    modelos.append(modelo)
    print(f"✅ Cargado: semilla {semilla}")

print(f"\nTotal: {len(modelos)} modelos")


# ============================================================
# PASO 2: CARGAR DATOS DE KAGGLE
# ============================================================
print("\n" + "="*60)
print("PASO 2: CARGAR DATOS KAGGLE")
print("="*60)

df_kaggle = pd.read_csv(RUTAS['kaggle'])
clientes = df_kaggle['numero_de_cliente'].values

X_kaggle, _, _ = preparar_para_modelo(df_kaggle)

print(f"Datos: {X_kaggle.shape}")
print(f"Clientes: {len(clientes):,}")


# ============================================================
# PASO 3: PREDECIR
# ============================================================
print("\n" + "="*60)
print("PASO 3: GENERAR PREDICCIONES")
print("="*60)

probs = predecir_ensemble(modelos, X_kaggle)

print(f"Probabilidades generadas:")
print(f"  Min: {probs.min():.6f}")
print(f"  Max: {probs.max():.6f}")
print(f"  Media: {probs.mean():.6f}")


# ============================================================
# PASO 4: CREAR SUBMISSION
# ============================================================
print("\n" + "="*60)
print("PASO 4: CREAR SUBMISSION")
print("="*60)

# Decidir número de envíos
if N_ENVIOS is not None:
    n_envios_final = N_ENVIOS
    print(f"Usando N° envíos fijo: {n_envios_final:,}")
else:
    n_envios_final = (probs >= THRESHOLD).sum()
    print(f"Usando threshold {THRESHOLD}: {n_envios_final:,} envíos")

print(f"% de la base: {n_envios_final/len(probs)*100:.2f}%")

# Crear submission
output_path = Path('data/submissions')
output_path.mkdir(exist_ok=True)

crear_submission(
    probs, clientes, n_envios_final,
    output_path / f'kaggle_{NOMBRE_EXPERIMENTO}_n{n_envios_final}.csv'
)

# Guardar también las probabilidades
np.save(output_path / f'probs_{NOMBRE_EXPERIMENTO}.npy', probs)
print(f"✅ Probabilidades guardadas")

print("\n" + "="*60)
print("✅ LISTO PARA KAGGLE")
print("="*60)
print(f"\nArchivo: submissions/kaggle_{NOMBRE_EXPERIMENTO}_n{n_envios_final}.csv")