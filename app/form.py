import streamlit as st
from utils import custom_select_box, custom_number_input


def form_submission(df):
    with st.form('data-entry'):
        st.write("**Características del paciente**")
        col1, col2 = st.columns(2)
        with col1:
            race = custom_select_box('Raza', df, 'race')
        with col2:
            gender = custom_select_box('Género', df, 'gender')
        age = st.select_slider(label='Edad',
                               options=df.age.unique())

        st.write("**Detalles de ingresos hospitalarios, procedimientos y medicamentos**")
        col3, col4, col5 = st.columns(3)
        with col3:
            admission_type_id = custom_select_box('Admission Type ID',
                                                  df,
                                                  'admission_type_id')
        with col4:
            discharge_disposition_id = custom_select_box('Discharge Disposition ID',
                                                         df,
                                                         'discharge_disposition_id')
        with col5:
            admission_source_id = custom_select_box('Admission Source ID',
                                                    df,
                                                    'admission_source_id')

        time_in_hospital = custom_number_input('Número de días en el hospital', 1)
        col1, col2 = st.columns(2)
        with col1:
            num_lab_procedures = custom_number_input('Número de procedimientos de laboratorio', 0)
        with col2:
            num_procedures = custom_number_input('Número de procedimientos', 0)

        num_medications = custom_number_input('Número de medicamentos', 0)
        col6, col7, col8 = st.columns(3)
        with col6:
            number_outpatient = custom_number_input('Número de visitas ambulatorias', 0)
        with col7:
            number_emergency = custom_number_input('Número de visitas de urgencia', 0)
        with col8:
            number_inpatient = custom_number_input('Número de visitas hospitalarias', 0)
        number_diagnoses = custom_number_input('Número de diagnósticos', 1)
        col9, col10 = st.columns(2)
        with col9:
            change = st.radio("Cambio en medicamentos diabéticos", ["Ch", "No"])
        with col10:
            diabetesMed = st.radio("Tuvo prescripción de medicamentos diabéticos", ["Yes", "No"])

        st.write("**Medicamentos prescritos**")
        options_med = df.glyburide.unique()

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            glyburide = st.radio("Glyburide", options_med, key='glyburide')
        with col2:
            glyburide_metformin = st.radio("Glyburide-metformin", options_med, key='glyburide-metformin')
        with col3:
            rosiglitazone = st.radio("Rosiglitazone", options_med, key='rosiglitazone')
        with col4:
            glipizide = st.radio("Glipizide", options_med, key='glipizide')
        with col5:
            insulin = st.radio("Insulin", options_med, key='insulin')
        with col6:
            glimepiride = st.radio("Glimepiride", options_med, key='glimepiride')

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            acarbose = st.radio("Acarbose", options_med, key='acarbose')
        with col2:
            repaglinide = st.radio("Repaglinide", options_med, key='repaglinide')
        with col3:
            nateglinide = st.radio("Nateglinide", options_med, key='nateglinide')
        with col4:
            pioglitazone = st.radio("Pioglitazone", options_med, key='pioglitazone')
        with col5:
            metformin = st.radio("Metformin", options_med, key='metformin')

        st.write("**Categorías de los diagnósticos realizados**")
        categories = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes',
                      'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']

        col1, col2, col3 = st.columns(3)
        with col1:
            diag_1 = st.selectbox('Diagnóstico 1',
                                  categories,
                                  index=None,
                                  placeholder='Seleccione')
        with col2:
            diag_2 = st.selectbox('Diagnóstico 2',
                                  categories,
                                  index=None,
                                  placeholder='Seleccione')
        with col3:
            diag_3 = st.selectbox('Diagnóstico 3',
                                  categories,
                                  index=None,
                                  placeholder='Seleccione')

        submit = st.form_submit_button('Predecir')

    if submit:
        x_input = [race, gender, age, admission_type_id, discharge_disposition_id,
                   admission_source_id, time_in_hospital, num_lab_procedures, num_procedures,
                   num_medications, number_outpatient, number_emergency, number_inpatient,
                   number_diagnoses, change, diabetesMed, glyburide, glyburide_metformin,
                   rosiglitazone, glipizide, insulin, glimepiride, acarbose, repaglinide,
                   nateglinide, pioglitazone, metformin, 0, 0, 0, diag_1, diag_2, diag_3, 0]
        return x_input
