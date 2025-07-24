# Gobernanza de datos en un análisis de predicción binaria con XGBoost en AWS, usando Python y PySpark.

El siguiente procedimiento asegura que  el análisis de predicción binaria mediante XGBoost en AWS cumpla con buenas prácticas de calidad y gobernanza de datos.

## I. Aseguramiento de la calidad de los datos

En esta sección se abordan las prácticas esenciales para asegurar la calidad de los datos que utilizarás en tu análisis predictivo. La gestión de la calidad de los datos es el primer paso crítico en cualquier pipeline de ciencia de datos, ya que la precisión y la fiabilidad de los modelos dependen en gran medida de la integridad, limpieza y consistencia de los datos de entrada. Aquí aprenderás a eliminar registros duplicados, gestionar valores atípicos y nulos, y validar los datos utilizando herramientas especializadas. Esto te permitirá preparar un dataset robusto y confiable para los siguientes pasos del análisis.

### I.1. Limpieza y preprocesamiento de datos con Pandas

En este apartado se explica cómo realizar la limpieza y el preprocesamiento de los datos utilizando la biblioteca Pandas de Python. Este proceso incluye la detección y eliminación de registros duplicados, el tratamiento de valores atípicos (outliers) que pueden distorsionar el análisis, y el manejo de datos faltantes mediante su eliminación o imputación. Además, se asegura la consistencia de los tipos de datos para cada columna, lo que es esencial para evitar problemas en las etapas posteriores de modelado y análisis.

```
python

import pandas as pd  # Importa la librería pandas para manejo de datos

# Ejemplo: Load data from S3 (requires AWS credentials set up)
data_path = 's3://your-bucket/data.csv'                                 # Ruta del archivo de datos en S3
df = pd.read_csv(data_path)                                             # Carga el archivo CSV en un DataFrame de pandas
```

```
python

before = len(df)                                                        # Guarda el número de filas antes de eliminar duplicados
df = df.drop_duplicates()                                               # Elimina filas duplicadas
after = len(df)                                                         # Guarda el número de filas después de eliminar duplicados
print(f"Removed {before - after} duplicate records.")                   # Imprime cuántos duplicados fueron eliminados
```

```
python

print(df.isnull().sum())  # Muestra la cantidad de valores nulos por columna
```

```
python

# Drop rows with missing target
df = df.dropna(subset=['target_column'])                                # Elimina filas donde la columna objetivo está vacía

# Imputar valores faltantes (ejemplo: rellenar columnas numéricas con la mediana)

for col in df.select_dtypes(include='number'):                          # Itera sobre columnas numéricas
    df[col].fillna(df[col].median(), inplace=True)                      # Rellena valores nulos con la mediana
```

```
python

from scipy.stats import zscore  # Importa zscore para detectar outliers

numeric_cols = df.select_dtypes(include='number').columns               # Selecciona columnas numéricas
z_scores = df[numeric_cols].apply(zscore)                               # Calcula el z-score para cada columna numérica
df = df[(z_scores.abs() < 3).all(axis=1)]                               # Filtra filas donde todos los z-scores absolutos son menores que 3
```

```
python

for col in numeric_cols:  # Itera sobre columnas numéricas
    Q1 = df[col].quantile(0.25)  # Calcula el primer cuartil
    Q3 = df[col].quantile(0.75)  # Calcula el tercer cuartil
    IQR = Q3 - Q1  # Calcula el rango intercuartílico
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]  # Filtra outliers
```

```
python

# Asegurar los tipos de datos correctos

df['date_col'] = pd.to_datetime(df['date_col'])                         # Convierte la columna de fecha al tipo datetime
df['category_col'] = df['category_col'].astype('category')              # Convierte la columna categórica al tipo category
```

---

### I.2. Validación de datos mediante la biblioteca “Great Expectations” de Python


Aquí aprenderás a utilizar la librería Great Expectations para validar la calidad y estructura de los datos. La validación de datos consiste en definir y verificar expectativas sobre el contenido, formato y valores permitidos en cada columna, asegurando que los datos cumplen reglas de negocio y estándares de calidad antes de ser usados en el modelado. Esto incluye, por ejemplo, comprobar que no haya valores nulos en ciertas columnas, que los valores numéricos estén dentro de un rango lógico o que los datos de correo electrónico tengan el formato adecuado. Esta práctica ayuda a prevenir errores y garantiza la reproducibilidad y confiabilidad del proceso analítico.

```bash
great_expectations init                                                 # Inicializa Great Expectations en el proyecto
```

```bash
great_expectations suite new                                            # Crea un nuevo conjunto de expectativas
```

```
python

import great_expectations as ge                                         # Importa la librería Great Expectations

context = ge.get_context()                                              # Obtiene el contexto de validación
batch = context.sources.pandas_default.read_csv(data_path)              # Lee los datos usando el contexto de pandas
```

```
python

validator = context.get_validator(                                      # Crea un validador para los datos
    batch=batch,                                                        # Usa el lote de datos cargado
    expectation_suite_name="your_suite"                                 # Nombre del conjunto de expectativas
)

validator.expect_column_values_to_not_be_null('target_column')          # Espera que la columna objetivo no tenga nulos
validator.expect_column_values_to_be_between('age', min_value=18, max_value=99)  # Espera que 'age' esté entre 18 y 99
validator.expect_column_values_to_match_regex('email', r".+@.+\..+")    # Espera que 'email' cumpla con el patrón de correo
```

```
python

results = validator.validate()                                          # Ejecuta la validación de los datos
print(results)                                                          # Imprime los resultados de la validación
```

```bash
great_expectations docs build                                           # Genera documentación visual de los resultados

# Ver los documentos generados en el navegador
```

---

## II. Cumplimiento de los datos

Esta sección se centra en garantizar que el manejo y procesamiento de los datos cumplan con las normativas y regulaciones vigentes, como GDPR, HIPAA o PCI-DSS. El cumplimiento normativo implica proteger la privacidad de los usuarios, restringir el acceso a información sensible, y asegurar que los datos se gestionen de acuerdo con los derechos y expectativas de los titulares. Aquí aprenderás a identificar qué normativas aplican a tu caso, cómo minimizar y anonimizar datos sensibles, y cómo implementar mecanismos para respetar los derechos de los usuarios, como la eliminación o anonimización de su información bajo requerimiento legal.

### II.1. Verificaciones de cumplimiento normativo

En este apartado se describen las estrategias para verificar y asegurar que los datos y procesos cumplen con las regulaciones relevantes. Esto incluye identificar la normativa aplicable (por ejemplo, GDPR para protección de datos personales en la Unión Europea, HIPAA para datos de salud en Estados Unidos, PCI-DSS para datos de tarjetas de pago), aplicar técnicas de minimización y anonimización de datos sensibles, y registrar el consentimiento y acceso a datos de los usuarios. Además, se recomienda aprovechar servicios de AWS como KMS para cifrado y IAM para gestión de accesos, fortaleciendo la seguridad y la trazabilidad del sistema.

```
python

# Ejemplo: Enmascaramiento de información personal identificable (PII)

if 'ssn' in df.columns:                                                 # Si existe la columna de número de seguro social
    df['ssn'] = df['ssn'].apply(lambda x: str(x)[:2] + "****" if pd.notnull(x) else x)  # Enmascara el valor para privacidad
```

```
python

# Eliminar datos de usuario a petición

user_id_to_remove = "user123"                                           # ID del usuario a eliminar
df = df[df['user_id'] != user_id_to_remove]                             # Elimina filas del usuario solicitado
```

```
python

import logging  # Importa la librería de logging

logging.basicConfig(filename='data_access.log', level=logging.INFO)      # Configura el archivo y nivel de log
logging.info(f"Accessed user data for analysis at {datetime.utcnow()}")  # Registra acceso a datos de usuario
```

---

### II.2. Registros de auditoría y documentación

En esta parte se muestra cómo llevar un registro detallado y transparente de todas las operaciones realizadas sobre los datos, lo que es fundamental para auditorías de cumplimiento y para la reproducibilidad del análisis. Se explica cómo utilizar la biblioteca de logging de Python para registrar cada etapa del pipeline de procesamiento, cómo versionar tanto los datos como el código, y cómo generar documentación que detalle el flujo y la transformación de la información. Además, se muestra cómo integrar PySpark para escalar el procesamiento a grandes volúmenes de datos y cómo documentar cada paso del proceso para facilitar revisiones y auditorías externas.

```
python

import logging  # Importa logging para auditoría
from datetime import datetime                                           # Importa datetime para marcar tiempo

logging.basicConfig(filename='pipeline_audit.log', level=logging.INFO)  # Configura archivo de auditoría

def log_step(step_desc):                                                # Define una función para registrar pasos
    logging.info(f"{datetime.utcnow().isoformat()} - {step_desc}")      # Registra la descripción con fecha/hora

log_step("Loaded data from S3.")                                        # Registra carga de datos
log_step("Removed duplicates.")                                         # Registra eliminación de duplicados
log_step("Imputed missing values.")                                     # Registra imputación de nulos
log_step("Performed outlier removal.")                                  # Registra eliminación de outliers
```

- Store scripts and configs in **Git**.                                 # Guarda código y configuración en Git
- Track data versions using tools like **DVC** or with S3 versioning.   # Versiona datos con DVC o S3

```
python

from pyspark.sql import SparkSession                                    # Importa SparkSession para PySpark

spark = SparkSession.builder.appName('DataGovernance').getOrCreate()    # Inicia Spark con nombre de la aplicación
df_spark = spark.read.csv('s3://your-bucket/data.csv', header=True, inferSchema=True)  # Lee datos desde S3 en Spark

log_step("Loaded data into Spark DataFrame")                            # Registra carga en Spark

# Convertir a Pandas para su posterior procesamiento si es necesario

df = df_spark.toPandas()                                                # Convierte DataFrame Spark a pandas
log_step("Converted Spark DataFrame to pandas DataFrame")               # Registra conversión
```

- Generate **Markdown** or **HTML** audit reports.                      # Genera reportes de auditoría en Markdown o HTML
- Use tools like **Jupyter Notebooks** for reproducibility and sharing  # Usa Jupyter para reproducibilidad y compartir

---



## III. Próximos pasos: Entrenamiento del modelo (ejemplo de XGBoost)

Una vez que los datos han sido limpiados, validados y cumplen con la normativa, puedes proceder al entrenamiento del modelo. En este apartado se muestra cómo preparar los datos para el entrenamiento, dividiendo en conjuntos de entrenamiento y prueba, y cómo inicializar y ajustar un modelo XGBoost de clasificación binaria. Este paso es esencial para obtener predicciones precisas y confiables, y debe realizarse únicamente después de asegurar la calidad y el cumplimiento de los datos.

```
python

import xgboost as xgb                                                   # Importa la librería XGBoost
from sklearn.model_selection import train_test_split                    # Importa función para dividir datos
 
X = df.drop('target_column', axis=1)                                    # Separa las características (X) eliminando la columna objetivo
y = df['target_column']                                                 # Selecciona la columna objetivo (y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Divide los datos en entrenamiento y prueba
model = xgb.XGBClassifier()                                             # Crea un clasificador XGBoost
model.fit(X_train, y_train)                                             # Entrena el modelo con los datos de entrenamiento
```

---

## Referencias

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [AWS Data Governance Best Practices](https://aws.amazon.com/architecture/data-governance-lake-house/)
- [GDPR Official Text](https://gdpr.eu/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/latest/python/python_api.html)

---
