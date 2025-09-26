import polars as pl
import json

# 1. Cargar el schema
with open("schema.json") as f:
    schema_dict = json.load(f)

# 2. Convertir a dtypes polars
schema = {k: getattr(pl, v) for k, v in schema_dict.items()}

# 3. Leer el CSV directamente, sin inferir
df = pl.read_csv("competencia_01_crudo.csv", schema_overrides=schema)

# 4. Función para agregar la columna clase_ternaria
def agregar_clase_ternaria_optimizada(df: pl.DataFrame) -> pl.DataFrame:
    """
    Versión optimizada usando Polars
    """
    return (
        df.lazy()
        .sort(["numero_de_cliente", "foto_mes"])
        .with_columns([
            pl.when(
                pl.col("foto_mes").shift(-1).over("numero_de_cliente").is_null() &
                (pl.col("foto_mes") != pl.col("foto_mes").max())
            ).then(pl.lit("BAJA+1"))
            .when(
                pl.col("foto_mes").shift(-2).over("numero_de_cliente").is_null() &
                (pl.col("foto_mes") <= pl.col("foto_mes").max() - 2)
            ).then(pl.lit("BAJA+2"))
            .otherwise(pl.lit("CONTINUA"))
            .alias("clase_ternaria")
        ])
        .collect()
    )

# 5. Agregar la columna clase_ternaria al dataframe
df = agregar_clase_ternaria_optimizada(df)

# 6. Tabla de frecuencia de clase_ternaria por foto_mes
tabla = (
    df.pivot(
        values="numero_de_cliente",
        index="foto_mes",
        on="clase_ternaria",
        aggregate_function="len"
    )
    .fill_null(0)  # reemplazar nulls por 0
)

print(tabla)
# 7. Guardar el dataframe resultante en un nuevo CSV
df.write_csv("competencia_01_con_clase_ternaria.csv")


