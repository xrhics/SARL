# SARL
A Spatial and Adversarial Representation Learning Approach for Land Use Classification with POIs.

## Data

* In this study we completely rely on open data, both in terms of the POI datasets and the land use datasets (as ground truth), to foster the reproducibility of the study. Here, we carry out thorough experiments in four major European cities: Amsterdam, Barcelona, Lisbon and Milan.
* We harvest POIs from Foursquare in our study areas (the four European cities) in 2018.  POI data can be crawled through Foursquare's API: https://api.foursquare.com/v2/venues/search.
* The POI category data can be downloaded at this link: https://developer.foursquare.com/docs/resources/categories.
* We utilize the Urban Atlas datasets as the ground truth for land use classification. url: https://land.copernicus.eu/local/urban-atlas.

## Data preprocessing

* All files are used for data preprocessing.
## model 

* keras_topK.py is used to pre-train the semantic pattern extractor. 
* Only after all data processing is complete can mian.py be run successfully. 

