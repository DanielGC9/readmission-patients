# Predicción de Readmisión de Pacientes en un Hospital


## Contenido

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Uso](#uso)


## Descripción

El objetivo de este proyecto es predecir la readmisión de pacientes en un hospital basado en diversos factores y características de los pacientes. Se han llevado a cabo los siguientes pasos:

1. **Limpieza de Datos**: Se han limpiado los datos para eliminar inconsistencias y valores nulos.
2. **Feature Engineering**: Se han creado nuevas variables y características que podrían influir en la predicción de readmisión.
3. **Entrenamiento de Modelos**: Se han entrenado dos modelos, CatBoost y LGBM, para realizar las predicciones.
4. **Desarrollo de la App**: Se ha creado una aplicación con Streamlit para facilitar la interacción con los modelos y visualizar las predicciones.

## Instalación

Para ejecutar este proyecto localmente, sigue los siguientes pasos:

1. Clona el repositorio:
    ```bash
    git clone https://github.com/DanielGC9/readmission-patients.git
    cd readmission-patients
    ```

2. Crea un entorno virtual e instala las dependencias:
    ```bash
    python -m venv venv
    source venv/bin/activate   # En Windows, usa `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Uso

Dentro de la carpeta notebooks se encuentra el notebook en el cual se realiza una explicación de lo realizado.

Para ejecutar la aplicación de Streamlit, utiliza el siguiente comando:
```bash
streamlit run app/app.py
```
Para ejecutar el modelo usa este comando
```bash
python src/main/main_cron.py
```

## Ejecución del Script Principal (`main_cron.py`)

El script `main_cron.py` contiene el código principal para la ejecución completa del modelo de predicción de readmisión de pacientes. Este script realiza varios pasos secuenciales:

Primero, se configuran las rutas necesarias y se inicia el proceso con mensajes de log. Luego, se lee el dataset original desde un archivo CSV (`data/diabetic_data.csv`).

A continuación, se llevan a cabo verificaciones de calidad de los datos mediante la función `data_qa`, y los datos limpios resultantes se almacenan en `data/data_clean.csv`.

Después, se realiza la ingeniería de características sobre los datos limpios utilizando la función `feature_eng`, y los nuevos datos generados se guardan en `data/new_data.csv`.

Posteriormente, se generan gráficos a partir de los nuevos datos usando la función `graphics`, y se guardan en la carpeta `data/images`.

Finalmente, el script entrena los modelos CatBoost y LightGBM utilizando los datos con las nuevas características a través de las funciones `model_catboost` y `model_lightgbm`. Los resultados del entrenamiento incluyen dos imágenes con métricas útiles que se almacenan en `data/metrics`. Además, los modelos entrenados se guardan en la carpeta `models`.

