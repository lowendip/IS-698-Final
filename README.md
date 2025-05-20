Steps To Run On Google Colab
1. Use this link to access the Google Colab notebook: https://colab.research.google.com/drive/1o9g8qVEOKmVZAJfYczGMvKcNDj9stlV3?usp=sharing
2. Upload all the files in this directory to the Google Colab temporary directory
3. Run the first cell to install dependencies
3. Download the dataset from (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
3. Upload the zip file to the Google Colab temporary directory (this will take a while)
4. Run the unzip cell corresponding to which version of the zip file you uploaded (archive.zip or chest_xray.zip)
5. If you uploaded archive.zip make sure to move the nested chest_xray folder out of the original folder (otherwise you will get an error)
6. Run the last cell (if you have issues with a label error try reloading the session or switching to GPU)

Steps To Run Locally
<<<<<<< HEAD
1. Download the dataset from (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and unzip it. 
2. Extract the chest_xray folder from the unzipped directory and place it in this directory 
3. Make sure to move the nested chest_xray folder out of the original folder
4. Make a virtual environment with Python 3.9
5. Pip install the requirements
6. Run Server.py
=======
1. Download the dataset from (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and unzip it. Extract the chest_xray folder from the unzipped directory and place it in this directory.
2. Make a virtual environment with Python 3.9
3. Pip install the requirements using this command: pip install --force-reinstall --no-dependencies -r requirements.txt 
4. Run Server.py
>>>>>>> 082d95e495576f779e59f546e41a1ba0b35819d6
