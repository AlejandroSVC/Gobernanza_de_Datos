# Gobernanza de datos en un modelo XGBoost en AWS, usando Python, PySpark y ZenML

![banner_datagovernance](docs/assets/images/banner_datagovernance.jpg)

El siguiente procedimiento asegura que un análisis de predicción binaria mediante XGBoost en AWS cumpla con buenas prácticas de calidad y gobernanza de datos.

Este script cubre:

1. Monitoreo continuo del modelo (data/model drift).
2. Detección temprana de drift.
3. Explicabilidad y transparencia del modelo.
4. Planes de reentrenamiento gestionados.
5. Documentación del ciclo de vida conforme a gobernanza.
6. Herramientas y automatización con ZenML y AWS.

Requisitos previos:

    • AWS S3 y permisos configurados.
    • Spark y PySpark instalados.
    • XGBoost y SHAP instalados.
    • ZenML instalado y configurado.

##  Sección 1. Carga de datos y preparación con PySpark 
Esta sección carga los datos desde AWS S3, realiza la limpieza básica y el preprocesamiento necesario para el modelo.
```
from pyspark.sql import SparkSession  # Importar SparkSession para manejar datos distribuidos
from pyspark.sql.functions import col  # Importar función para manipulación de columnas
```
### Inicialización de Spark Session
```
spark = SparkSession.builder.appName("XGBoostGobernanza").getOrCreate()  # Crear sesión de Spark
```
### Cargar datos desde S3
```
data_path = "s3a://mi-bucket/datos/dataset.csv"  # Ruta de los datos en S3
df = spark.read.csv(data_path, header=True, inferSchema=True)  # Leer archivo CSV
```
### Selección y limpieza de columnas relevantes
```
feature_cols = [c for c in df.columns if c != "target"]  # Columnas de características
df = df.dropna()  # Eliminar filas con valores nulos
```
### Conversión a Pandas para XGBoost (si el dataset es manejable en memoria)
```
pandas_df = df.toPandas()  # Convertir DataFrame de Spark a Pandas
```
##  Sección 2. Entrenamiento inicial del modelo XGBoost 
Se entrena el modelo XGBoost y se guarda el modelo entrenado en S3 para versionado y auditoría.
```
import xgboost as xgb  # Importar XGBoost
import numpy as np  # Importar numpy

X = pandas_df[feature_cols].values  # Extraer características
y = pandas_df["target"].values  # Extraer variable objetivo

dtrain = xgb.DMatrix(X, label=y)  # Crear DMatrix para XGBoost

params = {
    "objective": "binary:logistic",  # Objetivo de clasificación binaria
    "eval_metric": "auc",  # Métrica para evaluar
    "max_depth": 6,  # Profundidad máxima
    "eta": 0.1,  # Tasa de aprendizaje
    "seed": 42  # Semilla para reproducibilidad
}
model = xgb.train(params, dtrain, num_boost_round=100)  # Entrenar modelo
```
### Guardar el modelo en S3
```
model.save_model("/tmp/modelo_xgb.json")  # Guardar modelo localmente
import boto3  # Importar boto3 para AWS
s3 = boto3.client('s3')  # Cliente de S3
s3.upload_file("/tmp/modelo_xgb.json", "mi-bucket", "modelos/modelo_xgb_v1.json")  # Subir modelo a S3
```
##  Sección 3. Monitoreo continuo y detección de drift 
Implementamos monitoreo automatizado para detectar cambios en los datos (data drift) y el rendimiento (model drift) usando ZenML.

### Inicializar ZenML y pipeline de monitoreo
```
from zenml.pipelines import pipeline  # Importar pipeline de ZenML
from zenml.steps import step  # Importar step de ZenML

@step
def ingest_data() -> np.ndarray:  # Paso para ingesta de datos nuevos
    # Aquí reemplazar por la lógica real de ingesta (por ejemplo, S3, base de datos)
    new_data = pandas_df.sample(frac=0.2, random_state=1)[feature_cols].values  # Ejemplo: muestreo
    return new_data

@step
def detect_drift(new_data: np.ndarray) -> bool:  # Paso para detección de drift
    from scipy.stats import ks_2samp  # Prueba estadística de Kolmogorov-Smirnov
    orig_sample = X  # Usar datos de entrenamiento originales
    drift_detected = False  # Inicializar variable de drift
    p_values = []
    for i in range(orig_sample.shape[1]):  # Iterar por cada feature
        stat, p = ks_2samp(orig_sample[:, i], new_data[:, i])  # Prueba KS
        p_values.append(p)
    if np.mean(np.array(p_values) < 0.05) > 0.2:  # Si más del 20% de las features tienen p<0.05, hay drift
        drift_detected = True
    return drift_detected

@pipeline
def drift_monitoring_pipeline():  # Definir pipeline de monitoreo
    new_data = ingest_data()  # Ejecutar ingesta
    drift = detect_drift(new_data)  # Ejecutar detección de drift
    # Aquí se pueden agregar alertas automáticas, logs, etc.
```
### Ejecución del pipeline de monitoreo
```
drift_monitoring_pipeline()  # Ejecutar monitoreo
```
##  Sección 4. Explicabilidad y transparencia (XAI con SHAP) 
Se utiliza SHAP para explicar las predicciones del modelo y mantener transparencia en la toma de decisiones.
```
import shap  # Importar SHAP para explicabilidad

explainer = shap.TreeExplainer(model)  # Crear explicador SHAP para XGBoost
shap_values = explainer.shap_values(X)  # Calcular valores SHAP
```
### Visualización y reporte de explicabilidad
```
shap.summary_plot(shap_values, X, feature_names=feature_cols)  # Gráfico resumen de importancia de variables
```
##  Sección 5. Plan de reentrenamiento y control de versiones 
Si se detecta drift o baja performance, el modelo debe ser reentrenado automáticamente, versionando y auditando cada ciclo.

### Función para reentrenamiento (se puede automatizar con ZenML o AWS Lambda)
```
def retrain_model(new_X, new_y):
    dtrain_new = xgb.DMatrix(new_X, label=new_y)  # Crear DMatrix con nuevos datos
    new_model = xgb.train(params, dtrain_new, num_boost_round=100)  # Reentrenar modelo
    new_model.save_model("/tmp/modelo_xgb_v2.json")  # Guardar nuevo modelo
    s3.upload_file("/tmp/modelo_xgb_v2.json", "mi-bucket", "modelos/modelo_xgb_v2.json")  # Subir nuevo modelo a S3
    # Documentar y auditar cambio
    with open("/tmp/model_audit.log", "a") as f:  # Abrir archivo de auditoría
        f.write("Modelo reentrenado el 2025-08-04 con drift detectado.\n")  # Registrar evento
```
### Ejemplo de trigger automático (lógica simplificada)
```
if detect_drift(X):  # Si se detecta drift, reentrenar
    retrain_model(X, y)  # Reentrenar con los datos disponibles
```
##  Sección 6. Documentación y gobernanza del ciclo de vida 
Se documenta cada etapa, acción y métrica del ciclo de vida del modelo para cumplir con la gobernanza.

### Documentación estructurada del ciclo de vida del modelo
```
import datetime  # Importar módulo para fecha y hora
from pathlib import Path  # Para manejar rutas

def documentar_evento(evento, detalles):
    log_dir = "/tmp/model_audit_docs/"  # Directorio de logs
    Path(log_dir).mkdir(parents=True, exist_ok=True)  # Crear directorio si no existe
    log_file = f"{log_dir}lifecycle_{datetime.date.today()}.log"  # Archivo de log por fecha
    with open(log_file, "a") as f:  # Abrir archivo
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp
        f.write(f"[{timestamp}] EVENTO: {evento} - DETALLES: {detalles}\n")  # Escribir evento

documentar_evento("Entrenamiento inicial", "Modelo XGBoost v1 entrenado y subido a S3.")  # Documentar entrenamiento
documentar_evento("Monitoreo", "Pipeline de monitoreo ejecutado. No se detectó drift.")  # Documentar monitoreo
```
##  Sección 7. Herramientas y automatización del ciclo de vida 
ZenML y AWS se utilizan para versionar, monitorear, alertar y automatizar el ciclo de vida del modelo.

## Ejemplo de integración de ZenML para versionado y control de pipelines
```
from zenml.repository import Repository  # Importar el repositorio de ZenML
repo = Repository()  # Inicializar repositorio ZenML
```
### Registrar y versionar modelo entrenado, pipeline y artefactos
```
repo.get_pipeline("drift_monitoring_pipeline").run()  # Ejecutar pipeline y registrar artefactos
```
#  CONCLUSIONES 
"""
- Un enfoque de gobernanza proactivo permite detectar desvíos y responder de manera oportuna para mantener la confiabilidad y precisión del modelo.
- La automatización y la trazabilidad son clave: ZenML facilita la gestión de versiones, monitoreo y documentación del ciclo de vida del modelo.
- El uso de SHAP aporta transparencia, permite cumplir regulaciones y genera confianza en los usuarios del modelo.
- La integración con AWS y PySpark permite escalar el pipeline y aprovechar recursos cloud para grandes volúmenes de datos.
- Recomiendo mantener actualizada la documentación y pipelines, y revisar periódicamente las políticas de reentrenamiento y alertas.

Fuentes:
- https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html
- https://xgboost.readthedocs.io/
- https://shap.readthedocs.io/
- https://docs.zenml.io/
- https://spark.apache.org/docs/latest/api/python/



