import polars as pl
from pathlib import Path
import os
import datetime
import logging
from datetime import datetime

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
SEMILLA = 550007

MES_TRAIN = 202102
MES_VALIDACION = 202103
MES_TEST = 202104
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000
# =====================

def main():
    logger.info("Inicio de ejecución")
    # Leer el CSV en Polars
    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error al leer el CSV: {e}")
        return

    # Mostrar las primeras filas y formato
    print(df.head())
    print(df.schema)

    logger.info(f"Entrenando con SEMILLA={SEMILLA}, TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}")
    logger.info("Ejecucion finalizada.")
if __name__ == "__main__":
    main()