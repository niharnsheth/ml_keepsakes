# Machine Learing methods for Surgical Assessment

## Data
The dataset was created on a custom inhouse-built simulator I built which mainly contains a physical manikin and a virtual 3D patient. The simulator records surgical actions performed on the physical manikin and virtually displays the motion in real time. The models are trained on kinematic time series data generated from EM motion sensors connected to surgical instruments which record position and orientation at a frequency of 50 Hz.  
The data generated is not publicily available, due to licensing issues. 
The recorded data is for two neonatal surgical procedures: 
1. Pericardiocentesis (PCS) 
2. Thoracentesis (TCS). 

PCS has two surgical actions:  
* ChloraPrep applicaiton  
* Needle insertion  

TCS has 4 surgicla actions:  
* ChloraPrep applicaiton  
* Anesthetization  
* Needle insertion  
* Catheterization  


<!-- There is a publically available dataset called JIGSAWS which contains surgical performance data in the form of kinematics and videos, that is made available here. The dataset contains 3 surgical tasks performed using the surgical robot called DaVinci. There are differences in features and labels between our dataset and JIGSAWS, so naturally the scripts need to be editted to train for this dataset, however the overall function remains similar. -->

The dataset was pooled from surgical performances of full-time surgeons, residents and fellows within the Pediatric Dept.
1. In total 20 participants took part in generating the PCS data which included 4 surgeons, 7 fellows and 9 residents
2. In total 11 participants took part in generating the TCS data where 3 surgeons, 5 fellows and 3 residents  

### 1. Features
The motion sensors provide Position and Orientation information at a frequency of 50 Hz. Additional features like linear and angular velocity are calculated from the 
The feature set is a combination of information provided by the motion sensors and values calculated from that information along with time stamps  

Postion: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; X | Y | Z  
Orientation: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; W | Qx | Qy | Qz  
Linear Velocity: &nbsp;&nbsp;Vx | Vy | Vz  
Angular Velocity: VQx | VQy | VQz  

### 2. Labels  

The surgical performance were graded using the OSATS grading method into three classes Expert, Intermediate and Novice. Based on the manual grading the following number of samples for each label are present for developing the model.  
  
![label_dis_pericardiocentesis](https://user-images.githubusercontent.com/19583897/227019887-cd3c8959-5f7f-4e69-ac4c-8f8e722f2b10.png)  
![label_dist_thoracentesis](https://user-images.githubusercontent.com/19583897/227020232-191375cf-0a91-4a14-b0a5-1bcd9b05e41c.png)

The script was written to observe the distribution of features in the dataset. Mainly to recognize the outliers and determine the thresholds (min, max) for normalization of the features. 

### 3. Processing
The collected data had outliers of two forms:
* from sensor readings far away from the surgical area of interest
* unprecendeted motion unrelated to the surgical task
The later could only be removed manually by replaying the recorded data and removing unrelated motions in the task. The traditional outliers were removed by determining value thresholds (min and max) for each task, as seen below by plotting the density of values for the each feature per surgical task. 
  
 ![image](https://user-images.githubusercontent.com/19583897/228271853-326b52bf-e3e0-425b-80ed-63a3055fa8bb.png) 
  
 Post processing the features were normalized.  
   
## Training and Testing
Currently contains scripts for 
1. Siamese Convolution Neural Network
2. 1D-CNN

Siamese Convolution Neural Network  
  
Results:
Accuracy of the trained SCNN models  

![Screenshot 2023-03-28 080918](https://user-images.githubusercontent.com/19583897/228233998-70953ae1-f919-485b-9c94-267e962f1b8c.png)  

Experiment on the effect of input size on the performance of SCNN model  

![Screenshot 2023-03-28 081629](https://user-images.githubusercontent.com/19583897/228233260-b04487fb-553d-41db-99e5-cf7a69e8bfb9.png)  

Note: All the scirpts perform data augmentation using sliding window method. 
