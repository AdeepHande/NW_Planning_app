# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:10:32 2022

@author: Adeep
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


# @st.cache(suppress_st_warning=True)
def app():
    df_t11 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier1/master_csv_1_tier1_mod1.csv')
    df_t12 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier1/master_csv_1_tier1_mod2.csv')
    df_t21 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier2/master_csv_1_tier2_mod1.csv')
    df_t22 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier2/master_csv_1_tier2_mod2.csv')

    df_t1 = pd.concat([df_t11, df_t12])
    df_t1 = reverse_geocoder(df_t1)

    df_t2 = pd.concat([df_t21, df_t22])
    df_t2 = reverse_geocoder(df_t2)

    # Reading the master csv file
    # Master_t1_t2_t3: geopandas.sjoin(t1, t2, t3)

    default_params = pd.DataFrame({'model': ['Number of Points', 'Radius (in Km)', 'Bandwidth (in Mbps)'],
                                   'High Bandwidth (m1)': ['30', '0.5', '2048'],
                                   'Medium Bandwidth (m2)': ['60', '0.5', '1024'],
                                   'Crossover Model (m3)': ['75', '0.5', ''],
                                   'Wireless Model (m4)': ['10', '4', '20'],
                                   })

    df_t123 = pd.read_csv('D:/NW_Planning/NWP_v2/data/master_t1_t2_t3.csv')
    gdf = gpd.read_file(r'D:\NW_Planning\NWP_v2\Shape_India_v2\India_AC.shp')
    wireline = pd.read_csv('D:/NW_Planning/NWP_v2/data/wireline_v2/Wireline_tier1_tier2_POPs_with_CITY.csv')
    wireless = pd.read_csv('D:/NW_Planning/NWP_v2/data/wireline_v2/Wireless_live_BTS_with_CITY.csv')
    handhole = pd.read_csv('D:/NW_Planning/NWP_v2/data/wireline_v2/Handhole_TCL_with_CITY.csv')

    st.title('AI Based Network Expansion Planning')

    ## Ask if they want to upload data?

    # Add a case where selected city is not in external_df
    # so that final_df is not null
    # else:

    #   info = st.info("We will proceed with the internal data on our end")
    #  time.sleep(2)
    # info.empty()
    # A list of options to choose from;
    # The first selectbox of the web application

    state_list = list(np.unique(df_t1['state'].astype(str)))

    s_type = st.sidebar.selectbox('Select the state of your choice', state_list)

    st.sidebar.write('You selected:', s_type)

    city_type = ['Tier 1', 'Tier 2', 'Rest of India']

    c_type = st.sidebar.selectbox('Select the type of city you want to cluster in', city_type)

    # Displays the city you selected

    st.sidebar.write('You selected:', c_type)

    # Boolean values are False unless stated otherwise
    file_len = len(df_t123)
    t1_all = False
    t2_all = False
    t3_city = None

    if c_type == 'Tier 1':

        # Add 'All' as the first option

        t1_list = ['All'] + list(np.unique(df_t1[df_t1['state'] == s_type]['CITY'].astype(str)))

        # Option to choose multiple cities at a time

        city = st.sidebar.multiselect('Select the Tier 1 city of your choice', t1_list)

        # Returns city as a list;

        if city == 'All':

            city_summ = ['T1', 'All']
            t1_all = True
            city = t1_list[1:]
            st.sidebar.write('You selected:', city)
        else:
            city_summ = ['T1', city]
            st.sidebar.write('You selected', city)

    elif c_type == 'Tier 2':

        t2_list = ['All'] + list(np.unique(df_t2[df_t2['state'] == s_type]['CITY'].astype(str)))

        city = st.sidebar.multiselect('Select the Tier 2 city of your choice', t2_list)

        if city == 'All':

            city_summ = ['T2', 'All']
            city = t2_list[1:]
            st.sidebar.write('You selected:', city)

        else:
            city_summ = ['T2', city]
            t2_all = False
            st.sidebar.write('You selected', city)

    else:

        # t3_list = ['All'] +
        df_t123['ST_NAME'] = df_t123['ST_NAME'].str.lower()
        t3_list = list(np.unique(df_t123[df_t123['ST_NAME'] == s_type]['AC_NAME'].astype(str)))

        city = st.sidebar.multiselect('Select among the list of Tier 3, Tier 4, Tier 5, and Tier 6 cities', t3_list)
        t3_city = city
        city_summ = ['ROI', city]
        # if city == 'All':

        #    city = t3_list[1:]
        #    st.write('You selected:', city)
        # else:

        st.sidebar.write('You selected', city)

        # Provide optional parameters
    st.sidebar.write('Default Parameters')
    st.sidebar.write(default_params.T)
    st.sidebar.write('Scroll to the right to see the default parameters')
    modify_bool = st.sidebar.checkbox('I want to customize the parameters for clustering')

    if modify_bool:

        p1 = st.sidebar.slider('Select the number of points for the higher bandwidth model (model 1)', 0, 100, 30,
                               step=5)
        p2 = st.sidebar.slider('Select the number of points for the medium bandwidth model (model 2)', 0, 100, 60,
                               step=5)
        p3 = st.sidebar.slider('Select the number of points for the lower bandwidth (Crossover) model (model 3)', 0,
                               100, 75, step=5)
        p4 = st.sidebar.slider('Select the number of points for the wireless model (model 4)', 0, 100, 10, step=5)

        r1 = st.sidebar.slider('Select the radius (in Km) for the higher bandwidth model (model 1)', 0.0, 10.0, 0.5,
                               step=0.5)
        r2 = st.sidebar.slider('Select the radius (in Km) for the medium bandwidth model (model 2)', 0.0, 10.0, 0.5,
                               step=0.5)
        r3 = st.sidebar.slider('Select the radius (in Km) for the lower bandwidth (Crossover) model (model 3)', 0.0,
                               10.0, 0.5, step=0.5)
        r4 = st.sidebar.slider('Select the radius (in Km) for the wireless model (model 4)', 0.0, 10.0, 4.0, step=0.5)

        b1 = st.sidebar.slider('Select the Bandwidth (in Mbps) for the higher bandwidth model (model 1)', 0, 10000,
                               2000, step=50)
        b2 = st.sidebar.slider('Select the Bandwidth (in Mbps) for the medium bandwidth model (model 2)', 0, 10000,
                               1000, step=50)
        b3 = None
        b4 = st.sidebar.slider('Select the Bandwidth (in Mbps) for the wireless model (model 4)', 0, 100, 20, step=5)

        new_params = pd.DataFrame({'model': ['Number of Points', 'Radius (in Km)', 'Bandwidth (in Mbps)'],
                                   'model 1': [str(p1), str(r1), str(b1)],
                                   'model 2': [str(p2), str(r2), str(b2)],
                                   'model 3': [str(p3), str(r3), str(b3)],
                                   'model 4': [str(p4), str(r4), str(b4)],
                                   })

        if str(0) in new_params.values:
            raise ValueError('The parameters cannot be set to zero. Please rerun the app with adequate parameters.')
        st.sidebar.write('New Parameters')
        st.sidebar.write(new_params.T)
        st.sidebar.write('Scroll to the right to see the Customized parameters')
        params = {
            'm1': [p1, r1, b1],
            'm2': [p2, r2, b2],
            'm3': [p3, r3, b3],
            'm4': [p4, r4, b4]
        }
    else:
        params = None

    # if float(0) in params.values():

    #    raise ValueError('The parameters cannot be set to zero. Please rerun the app with adequate parameters.')
    # param_bool = st.radio(
    # "Do you want to provide parameters to compute the clusters?",
    # ('No', 'Yes'))
    # if param_bool == 'Yes':

    # st.sidebar.subheader('Optional Parameters')
    # st.sidebar.text('Model 1')
    # m1 = st.sidebar.number_input('Enter the number of points')

    # Add a run button

    if st.sidebar.button('Generate Clusters'):

        if t1_all == True:

            final_data = compute_all(df_t123, t1_list[1:], params)

            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                    hover_name='name', hover_data=["Bandwidth"],
                                    color_continuous_scale=px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r": 1, "t": 1, "l": 1, "b": 1})
            fig.show()
            # st.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')

        elif t2_all == True:

            final_data = compute_all(df_t123, t2_list[1:], params)

            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                    hover_name='name', hover_data=["Bandwidth"],
                                    color_continuous_scale=px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r": 1, "t": 1, "l": 1, "b": 1})
            fig.show()
            # st.sidebar.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')

        elif t3_city is not None:
            final_clustered_data = []

            for city in t3_city:
                parent_city = np.unique(df_t123[df_t123['AC_NAME'] == city]['CITY'])

                # df = df_t123[(df_t123['CITY']==parent_city[0])&(df_t123['AC_NAME']==city)]

                cluster_data = cluster_model(df_t123, parent_city[0], city, params)

                final_clustered_data.append(cluster_data)
                info1 = st.info(f'We have clustered all data points in {city}')
                time.sleep(2)
                info1.empty()

            final_data = pd.concat(final_clustered_data)

            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                    hover_name='name', hover_data=["Bandwidth"],
                                    color_continuous_scale=px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r": 1, "t": 1, "l": 1, "b": 1})
            fig.show()

            ## Distance from wireline, wireless, and handhole pop

            # st.sidebar.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')

        elif t1_all == False and t2_all == False and t3_city is None and city is None:

            st.error(
                'You have not selected any city, Please select the city / cities of your choice and click on the **_Generate_ _Clusters_** Button')

        else:

            final_clustered_data = []

            for city in city:
                flag = 0
                ac_names = np.unique(df_t123[df_t123['CITY'] == city]['AC_NAME'].astype(str))
                for ac_name in ac_names:
                    df = df_t123[(df_t123['CITY'] == city) & (df_t123['AC_NAME'] == ac_name)]

                    print(f'Computing Max Diameter Clustering for city:{city}, AC:{ac_name} ')

                    clustered_data = cluster_model(df, city, ac_name, params)
                    print(f'Completed Max Diameter Clustering for city:{city}, AC:{ac_name}')

                    final_clustered_data.append(clustered_data)

                    # my_bar = st.progress(0)

                    # flag += int(100/len(ac_names))
                    # time.sleep(1)
                    # my_bar.progress(flag)
                    info1 = st.info(f'We have clustered all data points in {ac_name} of {city}')
                    time.sleep(2)
                    info1.empty()
                    info2 = st.info(f'{len(ac_names) - flag} cities left in {city}')
                    time.sleep(2)
                    info2.empty()
                    flag += 1
            final_data = pd.concat(final_clustered_data)

            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                    hover_name='name', hover_data=["Bandwidth"],
                                    color_continuous_scale=px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r": 1, "t": 1, "l": 1, "b": 1})
            fig.show()

            # st.write(final_data.drop(columns=['CITY_TYPE','cluster']).head())

        df_cluster = final_data[['lat', 'lon', 'cluster_name']]

        df_cluster_mean = df_cluster.groupby(['cluster_name']).mean()
        df_cluster_mean = df_cluster_mean.reset_index('cluster_name')
        df_wireline = df_wireless = df_handhole = df_cluster_mean

        # Distance from POP
        # optimize
        df_wireline = compute_nearest_pop('wireline', df_wireline, wireline)
        df_wireliess = compute_nearest_pop('wireless', df_wireless, wireless)
        df_handhole = compute_nearest_pop('handhole', df_handhole, handhole)

        # Final matrix with category
        merged_final = create_final_matrix_with_category(final_data, df_wireline, df_wireless, df_handhole)

        # Bandwidth Potential Per Cluster matrix
        bwp_pc_matrix = BW_potential_per_cluster_matrix(merged_final)
        # Have to add the case for T1/T2 or ROI (Pending)
        summary = generate_summary(city_summ, df_t123, final_data, external_data, bwp_pc_matrix)
        st.write('**Clustered Feasiblity Request data**')
        st.write(merged_final)

        # st.write('**Summary:**')
        # st.write(summary)

        st.write("**Bandwidth Potential Per Cluster matrix**")
        st.write(bwp_pc_matrix)

        # st.write(final_data.shape)
        # st.write(df.shape)
        # st.sidebar.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')

        # @st.cache

        # st.write(merged_final)
        zipObj = ZipFile("output.zip", "w")
        uniq_city = list(final_data['AC_NAME'].unique())

        if len(uniq_city) < 196:

            for city in uniq_city:
                dummy = final_data[final_data['AC_NAME'] == city]
                fig = px.scatter_mapbox(dummy, lat="lat", lon="lon", color="cluster_name",
                                        hover_name='name', hover_data=["Bandwidth"],
                                        color_continuous_scale=px.colors.cyclical.IceFire)
                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r": 1, "t": 1, "l": 1, "b": 1})

                html_plot = fig.to_html()

                zipObj.writestr(f'{city}.html', html_plot)

        else:

            html_plot = fig.to_html()
            zipObj.writestr(f'plot.html', html_plot)

        # if not 'type' in final_data.columns:

        #    final_data = convert_df(final_data)

        # else:

        # merged_final = merged_final.groupby(['type'=='user_input'])
        # final_data = final_data.groupby(['type'=='user_input'])
        # merged_final = merged_final[merged_final['type']=='user_input']
        # final_data = final_data[final_data['type']=='user_input']

        final_data = convert_df(final_data)
        merged_final = convert_df(merged_final)
        bwp_pc_matrix = convert_df(bwp_pc_matrix)
        # Add multiple files to the zip
        zipObj.writestr(f'merged_final.csv', merged_final)
        zipObj.writestr(f'final_data.csv', final_data)
        zipObj.writestr(f'bandwith_potential_per_cluster.csv', bwp_pc_matrix)

        # close the Zip File
        zipObj.close()
        st.sidebar.success(
            'The clusters have been generated. Please go ahead and download the clusters, along with the bandwith potential per cluster')

        ZipfileDotZip = "output.zip"

        with open(ZipfileDotZip, "rb") as f:

            # st.sidebar.download_button('Download', f, file_name=f'{ZipfileDotZip}')

            bytes = f.read()
            b64 = base64.b64encode(bytes).decode()
            href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}'>\
                Download\
            </a>"

        st.sidebar.markdown(href, unsafe_allow_html=True)
        # should store it in a specific path
        # st.sidebar.write(f'The file can be found at {os.getcwd()}\output.zip')

    # Checkbox

    # return final_data # Should add cm here

    
     