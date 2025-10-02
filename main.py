import polars as pl
from pathlib import Path
import os
import datetime
import logging
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier

# Configurar logging y logs con fecha
os.makedirs("logs", exist_ok=True)
fecha_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/ejecucion_{fecha_actual}.log"
handlers = [logging.FileHandler(f"{log_filename}", mode='w', encoding='utf-8'), 
              logging.StreamHandler()]
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s -%(name)s -%(lineno)d - %(message)s",
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Definir la ruta del CSV
base_path = Path(__file__).parent
csv_path = base_path / "data" / "competencia_01_con_clase_ternaria.csv"
# =====================
# Configuración global
# =====================
SEMILLAS = [550007, 550019, 550031, 550033, 550047]

MES_TRAIN = 202102
MES_VALIDACION = 202103
MES_TEST = 202104
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000
# =====================

#Cargar datos funcion
def cargar_datos(path: str)-> pl.DataFrame | None :
    try:
        df = pl.read_csv(path)
        logger.info(f"Datos cargados correctamente desde {path}")
        return df
    except Exception as e:
        logger.error(f"Error al cargar datos desde {path}: {e}")
        return None

def main():
    logger.info("Inicio de ejecución")
    # Leer el CSV en Polars
    path = csv_path
    df = cargar_datos(path)
    if df is None:
        logger.error("Error al leer el CSV.")
        return

    # Mostrar las primeras filas y formato
    print(df.head())
    print(df.schema)

    logger.info(f"Entrenando con SEMILLA={SEMILLAS}, TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}")
    logger.info("Ejecucion finalizada.")
    return df   # ✅ ahora devolvemos el dataframe

if __name__ == "__main__":
    df = main()   # ✅ guardás df acá

## Fuego contra fuego.
#Febrero como entrenamiento, abril como test
X = df[df['foto_mes'] == MES_TRAIN]
y = X['clase_ternaria']
X = X.drop(columns=['clase_ternaria'])

X_futuro = df[df['foto_mes'] == MES_TEST]
y_futuro = X_futuro['clase_ternaria']
X_futuro = X_futuro.drop(columns=['clase_ternaria'])

# función de ganancia, para poder ser utilizada de una forma más genérica
def ganancia_prob(y_hat, y, prop=1, class_index=1, threshold=0.025):
  @np.vectorize
  def ganancia_row(predicted, actual, threshold=0.025):
    return  (predicted >= threshold) * (GANANCIA_ACIERTO if actual == "BAJA+2" else -COSTO_ESTIMULO)

  return ganancia_row(y_hat[:,class_index], y).sum() / prop

#Parametros optimizados
param_opt = {'criterion': 'entropy',
             'max_depth': 20,
             'min_samples_split': 145,
             'min_samples_leaf': 14,
             'max_leaf_nodes': 13}

model_opt = DecisionTreeClassifier(random_state=SEMILLAS[0], **param_opt)

model_opt.fit(X, y)
y_pred_opt = model_opt.predict_proba(X_futuro)
print(f"Ganancia de modelo Opt: {ganancia_prob(y_pred_opt, y_futuro)}")