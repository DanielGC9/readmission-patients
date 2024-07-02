import streamlit as st
import pandas as pd
import numpy as np
from utils import load_model, col_to_int
from form import form_submission
import sys
sys.path.append('.')
from src.main.utils.utils_functions import simplification_variables, count_medications, count_unique_diagnoses


# Features
features = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
            'num_procedures', 'num_medications', 'number_outpatient',
            'number_emergency', 'number_inpatient', 'number_diagnoses',
            'change', 'diabetesMed', 'glyburide', 'glyburide-metformin',
            'rosiglitazone', 'glipizide', 'insulin', 'glimepiride',
            'acarbose', 'repaglinide', 'nateglinide', 'pioglitazone', 'metformin',
            'precentageVisitsByAge', 'distinctDiagnoses', 'numberOfMedications',
            'categories_diag_1', 'categories_diag_2',
            'categories_diag_3', 'total_visits']

cat_features = ['race', 'admission_type_id', 'discharge_disposition_id',
                'admission_source_id', 'categories_diag_1', 'categories_diag_2',
                'categories_diag_3']
num_features = [feat for feat in features if feat not in cat_features]

df_diabetes = pd.read_csv("data/data_clean.csv")

# Encabezado de la página
st.markdown("""
    <style>
    .title {
        color: #3A4B99;
        text-align: center;
        font-size: 36px;
    }
    </style>
    <h1 class="title">Readmisión hospitalaria en pacientes diabéticos</h1>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .small-font {
        font-size: 20px;
        text-align: justify;
    }
    </style>
    <h3 class="small-font">Ingresa aquí las características del paciente para determinar bajo cierta probabilidad si el paciente puede o no ser readmitido al hospital en menos de 30 días</h3>
    """, unsafe_allow_html=True)


# Ingreso de datos a través del formulario
input = np.array(form_submission(df_diabetes))

# Procesamiento de datos
X_test = pd.DataFrame(columns=features)
X_test = pd.concat([X_test, pd.DataFrame([input], columns=features)],
                   ignore_index=True)

cat_features = X_test.select_dtypes('object').columns.tolist()
for col in cat_features:
    simplification_variables(X_test, col)

# Cambio de tipo de variable
for col in num_features:
    col_to_int(X_test, col)

X_test['precentageVisitsByAge'] = X_test['precentageVisitsByAge'].astype(float)

# Variables calculadas
# Number of unique diagnoses
X_test['distinctDiagnoses'] = X_test.apply(count_unique_diagnoses, axis=1)

# Number of medications used
X_test['numberOfMedications'] = X_test.apply(count_medications, axis=1)

# Percentage of number of visits by age
X_test['total_visits'] = X_test['number_inpatient'] + X_test['number_emergency'] + X_test['number_outpatient']
group = X_test.groupby('age')['total_visits'].mean()
X_test['precentageVisitsByAge'] = round(X_test['total_visits']/X_test['age'].map(group))

# LCargue del modelo
model = load_model('models/catboost_model.pkl')

# Predicción
y_pred = model.predict(X_test)
prob_pred = model.predict_proba(X_test)*100

if y_pred[0] == 1:
    st.info(f'El paciente tiene una probabilidad del **{prob_pred[0][1]:.1f}%** de **ser readmitido** en el hospital en **menos de 30 días.**', icon=":material/info:")
else:
    st.info(f'El paciente tiene una probabilidad del **{prob_pred[0][0]:.1f}%** de **no** ser readmitido en el hospital en **menos de 30 días.**', icon=":material/info:")
