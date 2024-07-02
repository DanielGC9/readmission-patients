import pickle
import streamlit as st


def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def custom_select_box(name, df, column):
    sb = st.selectbox(name,
                      df[column].unique(),
                      index=None,
                      placeholder='Seleccione')
    return sb


def custom_number_input(name, min):
    ni = st.number_input(label=name,
                         min_value=min,
                         value=None,
                         format='%d')
    return ni


def col_to_int(df, column):
    df[column] = df[column].astype(int)
