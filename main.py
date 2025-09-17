import polars as pl
from pathlib import Path

# Definir la ruta del CSV
base_path = Path(__file__).parent
csv_path = base_path / "competencia_01_con_clase_ternaria.csv"

# Leer el CSV en Polars
df = pl.read_csv(csv_path)

# Revisar primeras filas
print(df.head())


