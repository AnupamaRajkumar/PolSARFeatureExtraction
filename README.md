# ProcessChainForPolSARFeature
Hancrafted and shallow deep neural network for PolSAR feature extraction.\
Handcrafted features:
1. SAR polarimetric features - Pauli decomposition, Krogager decomposition, Freeman-Durden decomposition, Cloude and Pottier decomposition
2. Texture based features - GLCM and LBP
3. Color based features - MPEG-7
4. Morphological features - opening and closing
5. Shallow deep neural network - Autoencoder

# Results - Oberpfaffenhofen dataset

Original image - False color composite
![Original](https://github.com/AnupamaRajkumar/PolSARFeatureExtraction/blob/master/Artifacts/Original.png)

Labels 
![Labels](https://github.com/AnupamaRajkumar/PolSARFeatureExtraction/blob/master/Artifacts/LabelMap.png)

Classification result - KNN\
![Result](https://github.com/AnupamaRajkumar/PolSARFeatureExtraction/blob/master/Artifacts/FeatureExtractionResult.JPG)




