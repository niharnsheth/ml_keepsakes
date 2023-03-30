# [Parkinson's Disease Detection](https://github.com/niharnsheth/ml_keepsakes/tree/master/disease_prediction/parkinson_detection)

## Description

Parkinson's disease is a progressive disorder that affects the nervous system and the parts of the body controlled by the nerves. Symptoms start slowly. The first symptom may be a barely noticeable tremor in just one hand. Tremors are common, but the disorder may also cause stiffness or slowing of movement.

One of the major early symptom or sign of Parkinson's is changes in speach. Patients are found to speak softly, quickly, slur or hesistate before talking. The speech may be more monotone compared to the usual speech patterns.

Hence, this project uses a publicly available dataset created my Max Little of University of Oxford, in collaboration with the National Center for Voice and Speech, Colorado, who recorded speech signals. The dataset has a total of 23 features related to speech, with each sample classified. The labels are under the column "status"

The project uses some useful methods to remove data imbalance and reduce high diensionality. Post processing the following Machine Learning models are trained:
1. Linear Regression  
2. Decision Tree
3. Random Forest
4. Support Vector Machining
5. KNN
6. Gaussian Naive Bayes
7. XGBoost


# [Chronic Kidney Disease Detection](https://github.com/niharnsheth/ml_keepsakes/tree/master/disease_prediction/kidney_disease_detection)  
  
## Description

Chronic Kidney Disease (CKD) or chronic renal disease has become a major issue with a steady growth rate. A person can only survive without kidneys for an average time of 18 days, which makes a huge demand for a kidney transplant and Dialysis. Early detection is of utmost importance, which could be solved through ML.  

A public dataset containing 24 health related attributes taken in 2 month period of 400 patients. Out of the 400 records, 158 records have complete information the rest of the patient records are missing values.  

This work proposes a workflow to predict CKD status based on clinical data, incorporating data prepossessing, a missing value handling method with collaborative filtering and attributes selection.  

Currently, the project display the use of Fully Connected Neural Networks for classifying the presence of CKD.

