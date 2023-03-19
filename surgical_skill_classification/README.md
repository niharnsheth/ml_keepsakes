# Machine Learing methods for Surgical Assessment

## Dataset
The dataset was created on a custom inhouse-built simulator I built which mainly contains a physical manikin and a virtual 3D patient. The simulator records surgical actions performed on the physical manikin and virtually displays the motion in real time. The models are trained on kinematic time series data generated from EM motion sensors connected to surgical instruments which record position and orientation at a frequency of 50 Hz.  
The data generated is not publicily available, due to licensing issues. 
The recorded data is for two surgical procedures: 
1. Pericardiocentesis (PCS) 
2. Thoracentesis TCS). 

PCS has two surgical actions:  
a. Preping area of interest  
b. Needle Insertion  

TCS has 4 surgicla actions: #
a. Preping area of interest #
b. Anesthetization #
c. Needle Insertion #
d. Catheterization # 



There is a publically available dataset called JIGSAWS which contains surgical performance data in the form of kinematics and videos, that is made available here. The dataset contains 3 surgical tasks performed using the surgical robot called DaVinci. There are differences in features and labels between our dataset and JIGSAWS, so naturally the scripts need to be editted to train for this dataset, however the overall function remains similar. 

## Data Visualization
The script was writtent to observe the distribution of features in the dataset. Mainly to recognize the outliers and determine the thresholds (min, max) for normalization of the features. 

## Feature Engineering
The process entails the following:
1. The dataset was first manually cleaned to remove the parts of actions not related to surgical tasks.
2. Feature transformaion for orientation values to convert Euler Angles to Quaternions.
3. Additional features such as linear and rotational velocities were calculated through available time stamps for position and orientaiton values.
4. Normalization of data for training.

## Training and Testing
Currently contains scripts for 
1. 1D-CNN
2. LSTM
3. Siamese Convolution Neural Network
Note: All the scirpts perform data augmentation using sliding window method. 
