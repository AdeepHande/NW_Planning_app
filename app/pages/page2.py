# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 18:16:40 2022

@author: clouduser1
"""
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array
import matplotlib.pyplot as plt
from tqdm import tqdm
from geopandas import GeoSeries
import haversine as hs
import geopandas as gpd
import shapely.geometry
from shapely.geometry import Point, asPoint, Polygon, MultiPoint
import os
import json
import geog
import glob
import time
import reverse_geocoder as rg
from sklearn import metrics
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from tqdm import tqdm
import zipfile
import diameter_clustering as dc
from scipy.spatial.distance import pdist as pdist
from zipfile import ZipFile
import re
import plotly.express as px
import time
from zipfile import ZipFile
from io import StringIO, BytesIO
import base64
from utils import *


def app():
    df_t123 = pd.read_csv('D:/NW_Planning/NWP_v2/data/master_t1_t2_t3.csv')
    # gdf = gpd.read_file(r'D:\NW_Planning\NWP_v2\Shape_India_v2\India_AC.shp')
    wireline = pd.read_csv('D:/NW_Planning/NWP_v2/data/wireline_v2/Wireline_tier1_tier2_POPs_with_CITY.csv')
    wireless = pd.read_csv('D:/NW_Planning/NWP_v2/data/wireline_v2/Wireless_live_BTS_with_CITY.csv')
    handhole = pd.read_csv('D:/NW_Planning/NWP_v2/data/wireline_v2/Handhole_TCL_with_CITY.csv')

    st.write('A snippet of our internal data:')
    st.write(df_t123.head(5))
    external_data = st.radio(
        "Do you want to upload external data?",
        ('No', 'Yes'))

    # if "load_state" not in st.session_state:
    #    st.session_state.load_state = False

    if external_data == 'Yes':

        # Session State also supports attribute based syntax

        uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, key='external_data')
        # st.write('Number of Assembly Constitutencies in the internal data',len(np.unique(df_t123['AC_NAME'].astype(str))))

        if uploaded_file is not None:  # and 'flag' not in st.session_state:# or st.session_state.load_state:

            # st.session_state.flag = True
            # st.session_state.load_state = True
            # for uploaded_file in uploaded_files:
            external_df = pd.read_csv(uploaded_file)
            # st.write(dataframe)

            info = st.info('The file has been uploaded successfully')
            time.sleep(3)
            info.empty()
            info = st.info('Checking for duplicates after combining the data')
            time.sleep(3)
            info.empty()

            info.info('we will concatenate it with the internal data')
            time.sleep(3)
            info.empty()

            df_t123 = data_concat(df_t123, external_df, gdf)
            # st.write('Number of Assembly Constitutencies after uploading the external data',len(np.unique(df_t123['AC_NAME'].astype(str))))
            info.info('Dropping duplicates after combining the data')
            time.sleep(3)
            info.empty()
            # counter +=1

    st.title('AI Based Network Expansion Planning')

    line_pop = st.radio(
        "Do you want to upload the wireline point of presence (PoP) data?",
        ('No', 'Yes'))

    if line_pop == 'Yes':

        line_uploaded_data = st.file_uploader("Choose a CSV file", accept_multiple_files=False, key='wireline')

        if line_uploaded_data is not None:
            wireline = pd.read_csv(line_uploaded_data)

    less_pop = st.radio(
        "Do you want to upload the wireless point of presence (PoP) data?",
        ('No', 'Yes'))

    if less_pop == 'Yes':

        less_uploaded_data = st.file_uploader("Choose a CSV file", accept_multiple_files=False, key='wireless')

        if less_uploaded_data is not None:
            wireless = pd.read_csv(line_uploaded_data)

    hole_pop = st.radio(
        "Do you want to upload the Handhole point of presence (PoP) data?",
        ('No', 'Yes'))

    if hole_pop == 'Yes':

        hole_uploaded_data = st.file_uploader("Choose a CSV file", accept_multiple_files=False, key='handhole')

        if hole_uploaded_data is not None:
            handhole = pd.read_csv(line_uploaded_data)