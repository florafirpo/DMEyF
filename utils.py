"""
Funciones útiles reutilizables
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import gc
from config import PESOS, GANANCIA_ACIERTO, COSTO_ESTIMULO


def cargar_datos(path):
    """Carga CSV y agrega pesos"""
    df = pd.read_csv(path)
    
    # Agregar pesos
    df['clase_peso'] = PESOS['CONTINUA']
    df.loc[df['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = PESOS['BAJA+2']
    df.loc[df['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = PESOS['BAJA+1']
    
    return df


def preparar_para_modelo(df):
    """Convierte DataFrame a arrays para LightGBM"""
    # Separar features, target y pesos
    X = df.drop(['clase_ternaria', 'clase_peso', 'numero_de_cliente'], 
                axis=1, errors='ignore').values.astype('float32')
    y = (df['clase_ternaria'] != 'CONTINUA').astype('int8').values
    pesos = df['clase_peso'].values.astype('float32')
    
    return X, y, pesos


def metrica_ganancia(y_pred, data):
    """Métrica personalizada de ganancia para LightGBM"""
    pesos = data.get_weight()
    ganancia = np.where(pesos == PESOS['BAJA+2'], GANANCIA_ACIERTO, 0) - COSTO_ESTIMULO
    ganancia = ganancia[np.argsort(-y_pred)]
    return 'ganancia', np.max(np.cumsum(ganancia)), True


def entrenar_modelo(X, y, pesos, params, num_iterations):
    """Entrena un modelo LightGBM"""
    dataset = lgb.Dataset(X, label=y, weight=pesos)
    
    modelo = lgb.train(
        params,
        dataset,
        num_boost_round=num_iterations,
        feval=metrica_ganancia,
        callbacks=[lgb.log_evaluation(period=50)]
    )
    
    del dataset
    gc.collect()
    
    return modelo


def predecir_ensemble(modelos, X):
    """Promedia predicciones de varios modelos"""
    probs = np.mean([m.predict(X) for m in modelos], axis=0)
    return probs


def calcular_ganancia(probs, y_real, pesos):
    """Calcula ganancia ordenando por probabilidad"""
    indices = np.argsort(-probs)
    y_sorted = y_real[indices]
    pesos_sorted = pesos[indices]
    
    # Ganancia por cada registro
    ganancias = np.where(
        pesos_sorted == PESOS['BAJA+2'],
        GANANCIA_ACIERTO - COSTO_ESTIMULO,
        -COSTO_ESTIMULO
    )
    
    ganancia_acum = np.cumsum(ganancias)
    max_idx = np.argmax(ganancia_acum)
    
    return {
        'ganancia_maxima': float(ganancia_acum[max_idx]),
        'n_envios_optimo': int(max_idx + 1),
        'threshold_optimo': float(probs[indices[max_idx]]),
        'ganancia_acumulada': ganancia_acum
    }


def crear_submission(probs, clientes, n_envios, output_path):
    """Crea CSV para Kaggle"""
    indices_top = np.argsort(-probs)[:n_envios]
    pred = np.zeros(len(probs), dtype=int)
    pred[indices_top] = 1
    
    submission = pd.DataFrame({
        'numero_de_cliente': clientes,
        'Predicted': pred
    })
    
    submission.to_csv(output_path, index=False)
    print(f"✅ Submission guardado: {output_path}")
    
    return submission
