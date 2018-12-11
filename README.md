# Kickstarter Success Analysis

This is a personal project to analyze Kickstarter data to look for possible predictors of project launch success. 
It is hoped that a predictor algorithm can be put together that to some acceptable accuracy predicts whether future 
projects will successfully launch or not.

## Data
The raw Kickstarter data (the JSON file updated at 2018-10-18) was downloaded from: https://webrobots.io/kickstarter-datasets/. It 
is assumed that this data is accurate and no attempt was made to verify the web scraping tools used.

### Notes from raw data downloaded

**Note: from April 2015 we noticed that Kickstarter started limiting how many projects user can view in a single category. This limits 
the amount of historic projects we can get in a single scrape run. But recent and active projects are always included.**

*No attempt was made for this project to ensure we have the entirety of the project history.*

**Note: from December 2015 we modified the collection approach to go through all sub-categories instead of only top level categories. This 
yields more results in the datasets, but possible duplication where projects are listed in multiple categories. Also from December 2015 
JSON file is in JSON streaming format. Read more about it here: https://en.wikipedia.org/wiki/JSON_Streaming**

**We receive many question about timestamp format used in this dataset. It is unix time. Google has a lot of information about it.**

**Warning: files are compressed, size in area of 100mb. Uncompressed size around 600mb.**

*Due to Github size constraints, the raw dataset is not uploaded to this repository.*

## Goal
There are two potential goals of this project:

1. Analyze the raw data obtained to look for any trends and perhaps unlock some interesting visualizations regarding the different potential 
parameters to predict project launch success.

2. Build a prediction algorithm to try and predict whether future projects will successfully launch. 

## Instructions
Note that the working files are located in a 'working' directory. These need not be explored except out of interest, debugging, etc.

In order to run the model and make predictions on new Kickstarter data:

1. Ensure that Python is properly installed.
2. Ensure that the following files are located in the working directly:
	* classifier_rf_opt.pkl
	* f_cleanData.py
	* f_dataImport.py
	* f_predict.py
	* predict.py
	* sc_X.pkl
3. Create a folder named 'data' in the working directory.
4. Download or create the Kickstarter data to run the prediction model on.
	* The data must be in JSON format like from https://webrobots.io/kickstarter-datasets/.
	* Alternatively, the data can a comma-separated value dataframe from previously-run analyses.
5. Save the raw data in the 'data' folder.
6. Open a command prompt.
7. Type 'python predict.py'
8. Follow the prompts.