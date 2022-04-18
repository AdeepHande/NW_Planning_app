"""
Created on Thu Apr  7 21:51:04 2022

@author: Adeep Hande
"""

import streamlit as st
import pandas as pd
import reverse_geocoder as rg
from utils import *

import page1
import page2
import page3


st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 400px;
        margin-left: -400px;
    }

    """,
    unsafe_allow_html=True,
)

menu = """
<style>
#MainMenu {
  visibility: visible;
}
footer{
  visibility: visible;
}
footer:after{
    content:'Created @Tata Communications';
    display:block;
    position:relative;
    color:tomato;
}
</style>
"""
st.markdown(menu, unsafe_allow_html=True)

st.title('Welcome to the AI-based Network Expansion Planning App')
default_params = pd.DataFrame({'model': ['Number of Points', 'Radius (in Km)', 'Bandwidth (in Mbps)'],
                               'High Bandwidth (m1)': ['30', '0.5', '2048'],
                               'Medium Bandwidth (m2)': ['60', '0.5', '1024'],
                               'Crossover Model (m3)': ['75', '0.5', ''],
                               'Wireless Model (m4)': ['10', '4', '20'],
                               })

PAGES = {
    "Duplicate Checker": page1,
    "Upload external Data": page2,
    "AI Based Clustering App": page3
}

st.sidebar.title('AI Based Network Expansion Planning App')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()