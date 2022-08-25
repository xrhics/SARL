# SARL
A Spatial and Adversarial Representation Learning Approach for Land Use Classification with POIs.

## Data

* In this study we completely rely on open data, both in terms of the POI datasets and the land use datasets (as ground truth), to foster the reproducibility of the study. Here, we conducted extensive experiments in four major European cities: Amsterdam, Barcelona, Lisbon and Milan.
* We harvest POIs from Foursquare in our study areas (the four European cities) in 2018.  POI data can be crawled through Foursquare's API: https://api.foursquare.com/v2/venues/search.
* The POI category data can be downloaded at this link: https://developer.foursquare.com/docs/resources/categories.
* We utilize the Urban Atlas datasets as the ground truth for land use classification. url: https://land.copernicus.eu/local/urban-atlas.

## Data preprocessing

* Cities.py: In the source data, each building has a label. We divide the urban area into grids and calculate the proportion of each land use type in the grid based on the building area. We then assign the major land use type to each grid cell.
* Count_poi.py: The number of POIs for each POI category in each grid area is counted and stored.
* pre-data.py: The number of POIs contained in each category node in the zone semantic tree is calculated and stored.

## Model 

* keras_topK.py: keras_topK.py is used to pre-train the semantic pattern extractor. 
* grid.py: grid.py is utilized to process and output 3D tensors of the spatial configuration of regions.
* mydata.py: It handles the input and label information required for model training and testing.
* main.py: main.py can be run successfully once all data processing steps finish. 

