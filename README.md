### AI Based Network Expansion Planning App
The directory structure of the repository is as follows:

--------



    ├── README.md          
    ├── data
    │   ├── master.csv            <- Internal data comprising feasiblity requests from T1 and T2 cities.
    │   ├── PoP Data              <- Point of presence (PoP) data
    |   |   |── Wireline_pop.csv  
    |   |   |── Wireless_pop.csv
    |   |   └── handhole_pop.csv
    │   └── India_AC.shp          <- The shape files comprising all Assembly Constitutencies in India.
    │
    ├── models                    <- Parent model for implementing MaxDiameter Clustering Algorithm. 
    │   └── DC_clustering.py
    |
    ├── notebooks                 <- Jupyter notebooks  
    ├── requirements.txt          <- The requirements file for reproducing the analysis environment.
    │ 
    ├── src                       <- Source code for the app.
    │   ├── __init__.py           <- Makes src a Python module
    │   │
    │   ├── page1.py              <- Script to create Landing Page.  
    │   ├── page2.py              <- Script for Duplicate Checker (In dev) 
    │   ├── page3.py              <- Script for the clustering app. 
    |   ├── multipage.py          <- Script for the multipage (Docker Image built on the page)
    │   └── utils.py              <- Script for helper functions and utilities
    │
    ├── streamlit
    |   └── config.toml           <- configurations such as theme, fonts, etc.
    └── Dockerfile                <- Dockerfile to build the Docker Image and run it.
    


--------

The app is structured as follows:
* Page 1: Landing Page
  
      User has to select among
            * Duplicate Checker (Page 2)
            * Uploading external data (Page 3)
            * Clustering App (Page 4)
* Page 2: Duplicate Checker (In dev)

      Input: dataframe df to remove duplicates (Coordinates are necessary)
      Output: Returns dataframe df after checking for duplicates.
* Page 3: Uploading external data 

      Input: 
            A) Dataframe df (from Duplicate Checker)
            B) Wireline Point of Presence ( Wireline PoP)
            C) Wireless Point of Presence (Wireless PoP)
            D) Handhole Point of Presence ( Handhole PoP)
      Default data: 
            * Internal data (Master data)
      
      Output:
            * Spatially Join (Master Data, df) 🠆  df_dash
* Page 4: Clustering App
      
      Input data:
            * df_dash 
            * Configurable Parameters (set by the user)
      Returns:
            * Output.zip
                  ├── Final_clustered_data.csv
                  ├── Merged_final_data.csv  (categorized clustered data)
                  ├── Bandwidth_potential_per_cluster.csv
                  ├── plots.html 
                  └── plots.kml (In dev)

### Dev

1. Create an environment ```conda create -n streamlit-template python=3.8.5 pip```.

2. Install requirements: ```pip install -r app/requirements.txt```

#### Run the App
* ```cd app```
* ```streamlit run multipage.py```
