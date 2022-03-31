# -*- coding: utf-8 -*-
"""
random_forest_classification.py

Krishna Kumar Perikamana
30.03.2022 
https://www.researchgate.net/profile/Krishna-Kumar-Perikamana

Using random_forest_classification technique estimate the fractional built info at 30m scale from Landsat-8 image.
The same trained model can be used to estimate fractional built for a different region from Landsat-8 image.

"""

#import libraries
import numpy as np
from numpy import *
import pandas as pd
import random
from sklearn.feature_extraction import image
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from pyrsgis import raster
import seaborn as sns
import matplotlib.pyplot as plt

#Collect land_cover_training_data from the Landsat-8 image and labels from the stored file.
def fetch_land_cover_data(data_size,ws):
    data_path = "./Data/frac_landcover_built_2019.dat" #dataset_path
    column_names = ['B'] 
    data = pd.read_csv(data_path, skiprows = 0, names=column_names,
                          na_values = "?", comment='\t',
                          sep=" ", skipinitialspace=True)

    #drop  unknown values
    data.isna().sum()
    data = data.dropna()

    #Generating random numbers
    random.seed(1)
    rind = random.sample(range(len(data)), data_size)

    #Getting labels
    labels = data.iloc[rind]

    ds1,yourRaster = raster.read(r'''.\Data\190205_Bangalore_Landsat8_ra_30m_utm43n.tif''') #Landsat-8 multi spectral

    Z1 = yourRaster[1,:,:].flatten() #Blue
    Z2 = yourRaster[2,:,:].flatten() #Green
    Z3 = yourRaster[3,:,:].flatten() #Red
    Z4 = yourRaster[4,:,:].flatten() #NIR
    Z5 = yourRaster[5,:,:].flatten() #SWIR-1
    Z6 = yourRaster[6,:,:].flatten() #SWIR-2

    train_data=pd.DataFrame({
          'band1':Z1[rind],
          'band2':Z2[rind],
          'band3':Z3[rind],
          'band4':Z4[rind],
          'band5':Z5[rind],
          'band6':Z6[rind],
      })
    return train_data,labels


######Begining of main code######    

#Random Forest Classifier
train_data_size = 100000
test_data_size  = 2500
data_size = train_data_size + test_data_size
neighborhood_window_size = 2#e.g.if 1 takes 3x3 neighborhood,if 2 takes 5x5 neighborhood and so on
#compute actual window size
window_size = (neighborhood_window_size*2)+1

print('Getting Data ...')
LT_data,labels = fetch_land_cover_data(data_size,neighborhood_window_size)

#Generating random numbers
random.seed(0)
r1 = random.sample(range(len(LT_data)), train_data_size)
r2 = random.sample(range(len(LT_data)), test_data_size)

labels_dv = labels.values
train_labels = labels_dv[r1]
test_labels  = labels_dv[r2]

print('Training RF model ...')
#Build the model and train
model = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                  min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                  min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                  verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)


X_train_D = LT_data.iloc[r1]
X_train = X_train_D[['band1', 'band2', 'band3', 'band4', 'band5','band6']]  # Features
y_train = train_labels

X_test_D = LT_data.iloc[r2]
X_test = X_test_D[['band1', 'band2', 'band3', 'band4', 'band5','band6']]  # Features
y_test = test_labels

#fit model
model.fit(X_train,y_train)

print('Making predictions ...')
#Make predictions
y_pred_test = model.predict(X_test)
y_test_actual = y_test.ravel()

#compute RMSE values
rmse_pred = np.sqrt(metrics.mean_squared_error(y_test_actual.astype(int),y_pred_test.astype(int)))
print('%RMSE for prediction',rmse_pred)

print('Computing feature importance ...')
#compute feature importance
alpha =['Blue', 'Green', 'Red', 'NIR', 'SWIR1','SWIR2']
feature_imp = pd.Series(model.feature_importances_,index=alpha).sort_values(ascending=False)
print(feature_imp)

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

print('Making predictions for the whole image')
#Get full image data
dsLt,LT08 = raster.read(r'''.\Data\190205_Bangalore_Landsat8_ra_30m_utm43n.tif''') #Landsat-8 multi spectral

Z1 = LT08[1,:,:].flatten() #Blue
Z2 = LT08[2,:,:].flatten() #Green
Z3 = LT08[3,:,:].flatten() #Red
Z4 = LT08[4,:,:].flatten() #NIR
Z5 = LT08[5,:,:].flatten() #SWIR-1
Z6 = LT08[6,:,:].flatten() #SWIR-2

Ltimage_data=pd.DataFrame({
      'band1':Z1,
      'band2':Z2,
      'band3':Z3,
      'band4':Z4,
      'band5':Z5,
      'band6':Z6,
  })

#Make predictions for the whole image
built_predictions = model.predict(Ltimage_data)
print(built_predictions)
print(built_predictions.shape)

LT_frac_built = built_predictions.reshape(LT08.shape[1],LT08.shape[2])

plt.figure()
plt.imshow(LT_frac_built)
plt.show()

exportRaster_built = "./Data/190205_Bangalore_Landsat8_ra_30m_utm43n_Fractional_Built.tif"
raster.export(LT_frac_built, dsLt, exportRaster_built, dtype='uint8',compress='DEFLATE')
