# Machine Learing methods for Surgical Assessment

## Dataset
The dataset was created on a custom inhouse-built simulator I built which mainly contains a physical manikin and a virtual 3D patient. The simulator records surgical actions performed on the physical manikin and virtually displays the motion in real time. The models are trained on kinematic time series data generated from EM motion sensors connected to surgical instruments which record position and orientation at a frequency of 50 Hz.  
The data generated is not publicily available, due to licensing issues. 
The recorded data is for two neonatal surgical procedures: 
1. Pericardiocentesis (PCS) 
2. Thoracentesis TCS). 

PCS has two surgical actions:  
* ChloraPrep applicaiton  
* Needle insertion  

TCS has 4 surgicla actions:  
* ChloraPrep applicaiton  
* Anesthetization  
* Needle insertion  
* Catheterization  


<!-- There is a publically available dataset called JIGSAWS which contains surgical performance data in the form of kinematics and videos, that is made available here. The dataset contains 3 surgical tasks performed using the surgical robot called DaVinci. There are differences in features and labels between our dataset and JIGSAWS, so naturally the scripts need to be editted to train for this dataset, however the overall function remains similar. -->

## Data
The dataset was pooled from surgical performances of full-time surgeons, residents and fellows within the Pediatric Dept.
1. In total 20 participants took part in generating the PCS data which included 4 surgeons, 7 fellows and 9 residents
2. In total 11 participants took part in generating the TCS data where 3 surgeons, 5 fellows and 3 residents  

### Features
The motion sensors provide Position and Orientation information at a frequency of 50 Hz.
The feature set is a combination of information provided by the motion sensors and values calculated from that information along with time stamps  

Postion: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; X | Y | Z  
Orientation: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; W | Qx | Qy | Qz  
Linear Velocity: &nbsp;&nbsp;Vx | Vy | Vz  
Angular Velocity: VQx | VQy | VQz  

  
The surgical performance were graded using the OSATS grading method into three classes Expert, Intermediate and Novice.   
![label_dis_pericardiocentesis](https://user-images.githubusercontent.com/19583897/227019887-cd3c8959-5f7f-4e69-ac4c-8f8e722f2b10.png)  
![label_dist_thoracentesis](https://user-images.githubusercontent.com/19583897/227020232-191375cf-0a91-4a14-b0a5-1bcd9b05e41c.png)

The script was written to observe the distribution of features in the dataset. Mainly to recognize the outliers and determine the thresholds (min, max) for normalization of the features. 

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
