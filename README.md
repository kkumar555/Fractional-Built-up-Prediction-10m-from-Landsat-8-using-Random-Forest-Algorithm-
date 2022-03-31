# Fractional-Built-up-Prediction-10m-from-Landsat-8-using-Random-Forest-Algorithm-

Author: Krishna Kumar Perikamana / https://www.researchgate.net/profile/Krishna-Kumar-Perikamana / 03.2022

I am intrested in Computer vision, Image processing and Machine learning. If you use my code or some form of it in published work, please cite my GitHub repository:
If you use this code or some form of it in published work, please cite this repository: @misc{Fractional-Built-up-Prediction-10m, author = {Perikamana, K.K}, title = { Fractional-Built-up-Prediction-10m-from-Landsat-8-using-Random-Forest-Algorithm}, year = {2022}, publisher = {GitHub}, journal = {GitHub repository}, howpublished = {\url {https://github.com/kkumar555/Fractional-Built-up-Prediction-10m-from-Landsat-8-using-Random-Forest-Algorithm}} }

If you are interested on collaborating to do something interesting with this type of analysis...send me an email.

About this script:
This is a script that reads in Landsat-8 data, Esri Sentinel-2 10m land cover time series data and train a random forest classification algorithm to estimate fractional built cover at 30m scale. The trained model can be used to produce fractional land cover for other regions.
1.	You need Landsat-8 image for the same year and for the same extent along with Esri Sentinel-2 10m land cover time series data. [the ‘Data’ folder does not have Landsat-8 image which you need to download from the USGS website]
2.	First you need to run the script ‘compute_reference_fractional_built_data.py’ to compute fractional built-up from the Sentinel-2 10m land cover time series data. 
3.	Use this data to train a RF model along with Landsat-8 image. Now You can use this model to predict fractional built for other regions and other years if you have Landsat-8 image.

A sample output file is given which shows fractional built data for the city of Bangalore (India) for the year 2019.

<img width="895" alt="output" src="https://user-images.githubusercontent.com/102649572/161019702-84a416a8-e3d0-4b8c-b416-473749b0e4af.png">
