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

Para ejecutar la aplicación de Streamlit, utiliza el siguiente comando:
```bash
streamlit run app/app.py
