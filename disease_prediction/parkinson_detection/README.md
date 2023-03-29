## Data  
  
![Screenshot 2023-03-29 104323](https://user-images.githubusercontent.com/19583897/228575935-78b446dc-3b0b-4e28-bc37-0cd6efd72317.png)
  
 
### Label Distribution  
  
  
![image](https://user-images.githubusercontent.com/19583897/228573297-c476f1e1-7ca0-4e20-afba-0efd49e1045d.png)  
  
A large imbalance in number of samples per label can be observed from the plot above.
This can be fixed for tabular data by generating randomized samples through RandomOverSampler() from imbalanced-learn library for the label with fewer samples.  

### Feature Distribution

All the features can be observed through box plots and distribution plots. A sample of one feature is given below  
  
![image](https://user-images.githubusercontent.com/19583897/228574855-1fa7addb-43f4-49c2-9520-fd5e10eafd4b.png)  
    
![image](https://user-images.githubusercontent.com/19583897/228574994-6b7bd18e-52e3-42d4-a6e7-dbc13ea69fd3.png)

### Correlations

Plotting correlations between features and removing features which are highly correlated.
  
![image](https://user-images.githubusercontent.com/19583897/228576507-140c57fb-078b-442d-87dc-0baa60295f1f.png)

It is visible there are high correlations between features which can be reduced through dimensionality reduction using PCA.  
The features set is reduced from 22 to 8 after transforming the data. 

## Training and Results

The following ML models were trained on the data:  
1. Linear Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machining
5. KNN
6. Gaussian Naive Bayes
7. XGBoost
  
![image](https://user-images.githubusercontent.com/19583897/228580178-e0c3792e-dd19-4fae-b5d8-a73f8309a02a.png)

Most of the models perform well on testing. However, XGBoost and KNN stand out. 
1. KNN  
![image](https://user-images.githubusercontent.com/19583897/228580918-12b7af60-eb9a-4383-bb31-6120c627b2d7.png)  
  
2. XGBoost  
![image](https://user-images.githubusercontent.com/19583897/228581124-518d339a-2cf7-4390-bc0c-07897eab0757.png)




