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

# To enlarge the layout of the app, and fit it to the whole screen
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 600px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 400px;
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
) 


def main():
    
    # Reading the master csv file 
    # Master_t1_t2_t3: geopandas.sjoin(t1, t2, t3)
    
    df_t123 = pd.read_csv('D:/NW_Planning/NWP_v2/data/master_t1_t2_t3.csv')
    
    st.title('AI Based Network Expansion Planning')
    
    # A list of options to choose from; 
    # The first selectbox of the web application
    
    city_type = ['Tier 1', 'Tier 2', 'Rest of India']
    
    c_type = st.sidebar.selectbox('Select the type of city you want to cluster in', city_type) 
    
    # Displays the city you selected
    
    st.sidebar.write('You selected:', c_type)
    
    # Boolean values are False unless stated otherwise
    
    t1_all = False
    t2_all = False
    t3_city = None
   
    if c_type == 'Tier 1':
        
        # Add 'All' as the first option
        
        t1_list = ['All'] + list(np.unique(df_t1['CITY']))
        
        # Option to choose multiple cities at a time
        
        city = st.sidebar.multiselect('Select the Tier 1 city of your choice', t1_list)
        
        # Returns city as a list; 
        
        if city == 'All':
            t1_all = True
            city = t1_list[1:]
            st.sidebar.write('You selected:', city)
        else:
             
            st.sidebar.write('You selected', city)
            
    elif c_type == 'Tier 2':
        
        t2_list = ['All'] + list(np.unique(df_t2['CITY']))
        
        city = st.sidebar.multiselect('Select the Tier 2 city of your choice', t2_list)
        if city == 'All':
            
             
            city = t2_list[1:]
            st.sidebar.write('You selected:', city)
            
        else:
            t2_all = False
            st.sidebar.write('You selected', city)
    
    else:
        
        #t3_list = ['All'] + 
        t3_list = list(np.unique(df_t123['AC_NAME']))
        
        city = st.sidebar.multiselect('Select among the list of Tier 3, Tier 4, Tier 5, and Tier 6 cities', t3_list)
        t3_city = city
        
        #if city == 'All':
            
        #    city = t3_list[1:]
        #    st.write('You selected:', city)
        #else:
            
        st.sidebar.write('You selected', city)  
    
    ## Ask if they want to upload data?
    st.write('A snippet of our internal data:')
    st.write(df_t123.head(5))
    external_data = st.radio(
     "Do you want to upload external data?",
     ('No', 'Yes'))

     
        
     
    # Provide optional parameters
    st.sidebar.write('Default Parameters')
    st.sidebar.write(default_params)
    modify_bool = st.sidebar.checkbox('I want to customize the parameters for clustering')

    if modify_bool:
            
         
        p1 = st.sidebar.slider('Select the number of points for the higher bandwidth model (model 1)',0,100,30, step=5)
        p2 = st.sidebar.slider('Select the number of points for the medium bandwidth model (model 2)',0,100,60, step=5)
        p3 = st.sidebar.slider('Select the number of points for the lower bandwidth (Crossover) model (model 3)',0,100,75, step=5)
        p4 = st.sidebar.slider('Select the number of points for the wireless model (model 4)',0,100,10, step=5) 
        
        r1 = st.sidebar.slider('Select the radius (in Km) for the higher bandwidth model (model 1)',0.0, 10.0, 0.5, step=0.5)
        r2 = st.sidebar.slider('Select the radius (in Km) for the medium bandwidth model (model 2)',0.0, 10.0, 0.5, step=0.5)
        r3 = st.sidebar.slider('Select the radius (in Km) for the lower bandwidth (Crossover) model (model 3)',0.0, 10.0, 0.5, step=0.5)
        r4 = st.sidebar.slider('Select the radius (in Km) for the wireless model (model 4)',0.0, 10.0, 4.0, step=0.5)
        
        new_params = pd.DataFrame({'model':['Number of Points', 'Radius (in Km)'],
            'model 1': [str(p1), str(r1)],
            'model 2': [str(p2), str(r2)],
            'model 3': [str(p3), str(r3)],
            'model 4': [str(p4), str(r4)],
            })
            
        st.sidebar.write('New Parameters')
        st.sidebar.write(new_params)
        
        params = {
        'm1': [p1, r1],
        'm2': [p2, r2],
        'm3': [p3, r3],
        'm4': [p4, r4]
        }
    else:
        params = None
     
         
    #param_bool = st.radio(
    # "Do you want to provide parameters to compute the clusters?",
    # ('No', 'Yes'))
    #if param_bool == 'Yes':
        
    #st.sidebar.subheader('Optional Parameters')
    #st.sidebar.text('Model 1')
    #m1 = st.sidebar.number_input('Enter the number of points') 
    
    # Add a run button 
    
    if external_data == 'Yes':
        
        uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
        
        if uploaded_file is not None:
            #for uploaded_file in uploaded_files:
            external_df = pd.read_csv(uploaded_file)
            #st.write(dataframe)
        
            info = st.info('The file has been uploaded successfully')
            time.sleep(3)
            info.empty()
            info = st.info('Checking for duplicates after combining the data')
            time.sleep(3)
            info.empty()
            if list(external_df.columns) != list(df_t123.columns):
                
                st.error('The columns of the uploaded data and our internal data does not match')
                st.error('The columns of the uploaded data and our internal data does not match')
                st.write('Try to structure the external data in the following order', df_t123.columns)
            
            else:
                
                info.info('we will concatenate it with the internal data')
                time.sleep(3)
                info.empty()
                df = pd.concat([df_t123, external_df])
                df.drop_duplicates() 
                info.info('Dropping duplicates after combining the data')
                time.sleep(3)
                info.empty()
                
               
    #else:
        
     #   info = st.info("We will proceed with the internal data on our end")
     #  time.sleep(2)
     # info.empty()
        
    if st.sidebar.button('Generate Clusters'):
         
        if t1_all == True:
             
            final_data = compute_all(df_t123, t1_list[1:],params)
            
            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
            #fig.show()
            st.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')
            
        elif t2_all == True:
            
            final_data = compute_all(df_t123, t2_list[1:], params)
    
            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
            fig.show()
            st.sidebar.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')
            
        elif t3_city is not None:
            final_clustered_data = [] 
             
            for city in t3_city:
                    
                parent_city = np.unique(df_t123[df_t123['AC_NAME']==city]['CITY'])
                
                #df = df_t123[(df_t123['CITY']==parent_city[0])&(df_t123['AC_NAME']==city)]
                
                cluster_data = cluster_model(df_t123, parent_city[0], city, params)
            
                final_clustered_data.append(cluster_data)
                info1 = st.info(f'We have clustered all data points in {city}')
                time.sleep(2)
                info1.empty()
            final_data = pd.concat(final_clustered_data)
            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
            fig.show()     
            
            st.sidebar.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')
            
        elif t1_all == False and t2_all == False and t3_city is None and city is None:
            
            st.error('You have not selected any city, Please select the city / cities of your choice and click on the **_Generate_ _Clusters_** Button')
        
        else:
            
             
            final_clustered_data = []
            
            for city in city:
                flag = 0
                ac_names = np.unique(df_t123[df_t123['CITY']==city]['AC_NAME'])
                for ac_name in ac_names: 
                     
                    
                    df = df_t123[(df_t123['CITY']==city)&(df_t123['AC_NAME']==ac_name)] 
                
                    print(f'Computing Max Diameter Clustering for city:{city}, AC:{ac_name} ')
                     
                    
                    clustered_data = cluster_model(df, city, ac_name, params) 
                    print(f'Completed Max Diameter Clustering for city:{city}, AC:{ac_name}')
                    
                    final_clustered_data.append(clustered_data) 
                     
                    #my_bar = st.progress(0)
        
                    #flag += int(100/len(ac_names))
                    #time.sleep(1)
                    #my_bar.progress(flag)  
                    info1 = st.info(f'We have clustered all data points in {ac_name} of {city}')
                    time.sleep(2)
                    info1.empty()
                    info2 = st.info(f'{len(ac_names) - flag} cities left in {city}')
                    time.sleep(2)
                    info2.empty() 
                    flag += 1 
            final_data = pd.concat(final_clustered_data)
        
            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                    hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
            fig.show()
        
            st.sidebar.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')
            
        #st.sidebar.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')
        
         
    # Checkbox 
        zipObj = ZipFile("clusters.zip", "w")
        city_list = list(np.unique(final_data['AC_NAME']))
        
        if len(city_list) == 1:
            fig.to_html(f'{city}.html')
            zipObj.write(f'{city}.html')
            
        else:
             
             
            for city in city_list:
                city_cluster_data = final_data[final_data['AC_NAME']==city]
                fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                    hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
                plot = fig.to_html(f'{city}.html')
                zipObj.write(f'Plots/{plot}.html')
            
        zipObj.close()
        ZipfileDotZip = "clusters.zip" 
        
        with open(ZipfileDotZip, "rb") as f:
            bytes = f.read()
            b64 = base64.b64encode(bytes).decode()
            href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}.zip'>\
                City-wise clusters\
            </a>"
         
        
            st.download_button(
                 label="Download",
                 data=f,
                 file_name='clusters.zip',
                 mime='zip',
             )
            
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.sidebar.write('Click here to download the clusters and confusion matrices')
        
        

 
                 
    
    
    else:
        st.sidebar.write('Click here to generate the clusters and confusion matrices')
        
         
    
     

    #return final_data # Should add cm here

st.markdown('Created by **_Tata_ _Communications_**') 
 # cm as a table, summary of the final data all in the middle
 # 2 tables, 3 cs
    
if __name__ == "__main__":
     
     
    default_params = pd.DataFrame({'model':['Number of Points', 'Radius (in Km)'],
    'model 1': ['30', '0.5'],
    'model 2': ['60', '0.5'],
    'model 3': ['75', '0.5'],
    'model 4': ['10', '4'],
    }) 
    df_t11 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier1/master_csv_1_tier1_mod1.csv')
    df_t12 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier1/master_csv_1_tier1_mod2.csv')
    df_t21 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier2/master_csv_1_tier2_mod1.csv')
    df_t22 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier2/master_csv_1_tier2_mod2.csv')
    
    df_t1 = pd.concat([df_t11,df_t12])
    df_t2 = pd.concat([df_t21,df_t22])
     
    main()
    
     