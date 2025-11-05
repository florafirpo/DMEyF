"""
Script 1: Entrenar con 01,02,03 y evaluar en 04
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

from config import SEMILLAS, RUTAS, GANANCIA_ACIERTO, COSTO_ESTIMULO
from utils import (
    cargar_datos, preparar_para_modelo, entrenar_modelo,
    predecir_ensemble, calcular_ganancia, crear_submission
)


# ============================================================
# CONFIGURACIÓN
# ============================================================
ENTRENAR_NUEVO = True  # False para usar modelos ya entrenados
NOMBRE_EXPERIMENTO = "exp_01_02_03"


# ============================================================
# PASO 1: ENTRENAR CON 01, 02, 03
# ============================================================
print("\n" + "="*60)
print("PASO 1: ENTRENAMIENTO")
print("="*60)

if ENTRENAR_NUEVO:
    # Cargar hiperparámetros
    with open(RUTAS['hiperparametros'], 'r') as f:
        config = json.load(f)[0]  # Usar el mejor
    
    print(f"Hiperparámetros: Trial {config['trial_number']}")
    
    # Cargar y preparar datos
    print("Cargando datos de entrenamiento...")
    df_train = cargar_datos(RUTAS['train_123'])
    X_train, y_train, pesos_train = preparar_para_modelo(df_train)
    
    print(f"Datos: {X_train.shape}")
    print(f"BAJA+2: {(df_train['clase_ternaria'] == 'BAJA+2').sum():,}")
    
    # Preparar parámetros
    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'max_bin': 31,
        'num_leaves': config['params']['num_leaves'],
        'learning_rate': config['params']['learning_rate'],
        'min_data_in_leaf': config['params']['min_data_in_leaf'],
        'feature_fraction': config['params']['feature_fraction'],
        'bagging_fraction': config['params']['bagging_fraction'],
        'verbose': -1
    }
    
    # Entrenar un modelo por semilla
    modelos = []
    for i, semilla in enumerate(SEMILLAS):
        print(f"\nEntrenando modelo {i+1}/{len(SEMILLAS)} (semilla {semilla})...")
        params['seed'] = semilla
        
        modelo = entrenar_modelo(
            X_train, y_train, pesos_train,
            params, config.get('best_iter', 100)
        )
        
        modelos.append(modelo)
        
        # Guardar
        modelo.save_model(f'data/modelo_{NOMBRE_EXPERIMENTO}_semilla_{semilla}.txt')
    
    print(f"\n✅ {len(modelos)} modelos entrenados")
    
    # Limpiar memoria
    del df_train, X_train, y_train, pesos_train
    import gc
    gc.collect()

else:
    # Cargar modelos existentes
    print("Cargando modelos existentes...")
    import lightgbm as lgb
    
    modelos = []
    for semilla in SEMILLAS:
        path = f'data/modelo_{NOMBRE_EXPERIMENTO}_semilla_{semilla}.txt'
        modelo = lgb.Booster(model_file=path)
        modelos.append(modelo)
    
    print(f"✅ {len(modelos)} modelos cargados")


# ============================================================
# PASO 2: EVALUAR EN 04
# ============================================================
print("\n" + "="*60)
print("PASO 2: EVALUACIÓN EN TEST (04)")
print("="*60)

# Cargar datos de test
print("Cargando datos de test...")
df_test = pd.read_csv(RUTAS['test_04'])

# Necesitamos clase_ternaria original y pesos
df_test['clase_peso'] = 1.0
df_test.loc[df_test['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
df_test.loc[df_test['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

y_test_original = df_test['clase_ternaria'].values
pesos_test = df_test['clase_peso'].values
clientes_test = df_test['numero_de_cliente'].values

X_test, _, _ = preparar_para_modelo(df_test)

print(f"Datos test: {X_test.shape}")
print(f"BAJA+2: {(y_test_original == 'BAJA+2').sum():,}")

# Predecir
print("\nGenerando predicciones...")
probs = predecir_ensemble(modelos, X_test)

print(f"Probabilidades: min={probs.min():.4f}, max={probs.max():.4f}, media={probs.mean():.4f}")

# Calcular ganancia
print("\nCalculando ganancia...")
resultados = calcular_ganancia(probs, y_test_original, pesos_test)

print(f"\n🎯 RESULTADOS:")
print(f"  Ganancia máxima: ${resultados['ganancia_maxima']:,.0f}")
print(f"  N° envíos óptimo: {resultados['n_envios_optimo']:,}")
print(f"  Threshold óptimo: {resultados['threshold_optimo']:.6f}")
print(f"  % de la base: {resultados['n_envios_optimo']/len(probs)*100:.2f}%")

# Calcular captura de BAJA+2
indices_opt = np.argsort(-probs)[:resultados['n_envios_optimo']]
n_baja2_total = (y_test_original == 'BAJA+2').sum()
n_baja2_capturados = (y_test_original[indices_opt] == 'BAJA+2').sum()

print(f"\n📊 CAPTURA BAJA+2:")
print(f"  Total: {n_baja2_total:,}")
print(f"  Capturados: {n_baja2_capturados:,}")
print(f"  Tasa: {n_baja2_capturados/n_baja2_total*100:.1f}%")


# ============================================================
# PASO 3: GRÁFICOS
# ============================================================
print("\n" + "="*60)
print("PASO 3: GRÁFICOS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Evaluación: {NOMBRE_EXPERIMENTO}', fontsize=14, fontweight='bold')

# Gráfico 1: Ganancia acumulada
ax1 = axes[0, 0]
n_envios = np.arange(1, len(resultados['ganancia_acumulada']) + 1)
ax1.plot(n_envios, resultados['ganancia_acumulada']/1e6, 'b-', linewidth=2)
ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
ax1.axvline(resultados['n_envios_optimo'], color='g', linestyle='--', alpha=0.7)
ax1.set_xlabel('N° Envíos')
ax1.set_ylabel('Ganancia (Millones $)')
ax1.set_title('Ganancia Acumulada')
ax1.grid(alpha=0.3)

# Gráfico 2: Ganancia vs umbral
ax2 = axes[0, 1]
umbrales = np.linspace(0, 1, 500)
ganancias_umbral = []
for u in umbrales:
    n = (probs >= u).sum()
    if n > 0 and n <= len(resultados['ganancia_acumulada']):
        ganancias_umbral.append(resultados['ganancia_acumulada'][n-1])
    else:
        ganancias_umbral.append(0)

ax2.plot(umbrales, np.array(ganancias_umbral)/1e6, 'purple', linewidth=2)
ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
ax2.axvline(resultados['threshold_optimo'], color='g', linestyle='--', alpha=0.7)
ax2.set_xlabel('Umbral de Probabilidad')
ax2.set_ylabel('Ganancia (Millones $)')
ax2.set_title('Ganancia vs Umbral')
ax2.grid(alpha=0.3)

# Gráfico 3: Distribución por clase
ax3 = axes[1, 0]
prob_continua = probs[y_test_original == 'CONTINUA']
prob_baja1 = probs[y_test_original == 'BAJA+1']
prob_baja2 = probs[y_test_original == 'BAJA+2']

ax3.hist(prob_continua, bins=50, alpha=0.5, label='CONTINUA', color='blue')
ax3.hist(prob_baja1, bins=50, alpha=0.6, label='BAJA+1', color='orange')
ax3.hist(prob_baja2, bins=50, alpha=0.8, label='BAJA+2', color='red')
ax3.axvline(resultados['threshold_optimo'], color='g', linestyle='--', linewidth=2)
ax3.set_xlabel('Probabilidad')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Distribución por Clase')
ax3.legend()

# Gráfico 4: Ganancia por percentil
ax4 = axes[1, 1]
percentiles = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15]
gan_pct = []
for p in percentiles:
    n = int(len(probs) * p)
    if n > 0 and n <= len(resultados['ganancia_acumulada']):
        gan_pct.append(resultados['ganancia_acumulada'][n-1]/1e6)

labels = [f"{p*100:.1f}%" for p in percentiles[:len(gan_pct)]]
ax4.bar(labels, gan_pct, color='teal', alpha=0.8)
ax4.axhline(0, color='r', linestyle='--', alpha=0.5)
ax4.set_xlabel('Top % Clientes')
ax4.set_ylabel('Ganancia (Millones $)')
ax4.set_title('Ganancia por Percentil')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()

# Guardar
output_path = Path('data/evaluaciones')
output_path.mkdir(exist_ok=True)
plt.savefig(output_path / f'analisis_{NOMBRE_EXPERIMENTO}.png', dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: {output_path / f'analisis_{NOMBRE_EXPERIMENTO}.png'}")

plt.show()


# ============================================================
# PASO 4: GUARDAR RESULTADOS
# ============================================================
print("\n" + "="*60)
print("PASO 4: GUARDAR RESULTADOS")
print("="*60)

# Guardar métricas
metricas = {
    'experimento': NOMBRE_EXPERIMENTO,
    'ganancia_maxima': resultados['ganancia_maxima'],
    'n_envios_optimo': resultados['n_envios_optimo'],
    'threshold_optimo': resultados['threshold_optimo'],
    'n_baja2_capturados': int(n_baja2_capturados),
    'tasa_captura': float(n_baja2_capturados / n_baja2_total)
}

with open(output_path / f'metricas_{NOMBRE_EXPERIMENTO}.json', 'w') as f:
    json.dump(metricas, f, indent=4)

print(f"✅ Métricas guardadas")

# Guardar submission con punto óptimo
crear_submission(
    probs, clientes_test, 
    resultados['n_envios_optimo'],
    output_path / f'submission_{NOMBRE_EXPERIMENTO}.csv'
)

print("\n" + "="*60)
print("✅ PROCESO COMPLETADO")
print("="*60)