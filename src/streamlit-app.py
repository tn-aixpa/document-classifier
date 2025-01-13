
import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import digitalhub as dh
import os
import streamlit

def annotate(text, nl):
    body = {"inference_input": { "text": text, "k": nl} }
    preds = requests.post(f"http://{service_url}/", json=body).json()
    return preds


st.title('Document classifier')
correspondence = pd.read_csv("src/correspondence.csv")
service_url = os.environ.get("SERVICE_URL", "")

if service_url == None or service_url == "":
    service_url = st.text_input("Service Endpoint", value="", placeholder="host:port")

ta = st.text_area("Testo", value="", placeholder="Fornisci il testo da classificare", height=340)
nl = st.number_input("Numero di labels", min_value=1)

if st.button("Annota"):
    pred = annotate(ta, nl)
    #st.text(pred['results'])
    result = ''
    for el in pred['results']:
        dID, c, cd, m, md = correspondence.loc[
        correspondence.ID_tassonomia == el, ['azione', 'campi', 'descrizione_codice_campo', 'macroambiti', 'descrizione_codice_macro']].values[0]
        result = result + f'ID tassonomia di azione: {dID}' + '\n'
    st.text(result)
