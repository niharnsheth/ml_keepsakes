import tensorflow as tf
import pandas as pd

from tensorflow import keras
from keras import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import GlobalAvgPool1D, Flatten

class BaseInputLayerBuilder():
    def __init__(self, n_features, name=None, filters=3, kernel_size=8, activation='relu'):
        self.input = Input(shape=(n_features),name=name)
        self.x = Conv1D(filters=filters, 
                        kernel_size=kernel_size,
                        activation=activation)(self.input)
        self.x = MaxPool1D(4)(self.x)
        


class SecondaryInputLayerBuilder():

    def __init__(self):

        self.input_position = BaseInputLayerBuilder(n_features=3, name='position',
                                               filters=3, kernel_size=8,
                                               activation='relu').x
        

        self.input_linear_velocity = BaseInputLayerBuilder(n_features=3, name='linear_velocity',
                                                      filters=3, kernel_size=8,
                                                      activation='relu').x

        self.input_angular_velocity = BaseInputLayerBuilder(n_features=3, name='angular_velocity',
                                                       filters=3, kernel_size=8,
                                                       activation='relu').x
        
        self.input_rotation_matrix = BaseInputLayerBuilder(n_features=9, name='rotation_matrix',
                                                      filters=9, kernel_size=8,
                                                      activation='relu').x
        
        self.input_gripper_angular_velocity =  BaseInputLayerBuilder(n_features=1, name='gripper_angular_velocity',
                                                                filters=9, kernel_size=8,
                                                                activation='relu').x
        


class ModelBuilder():

    def __init__(self, n_features, n_classes, learning_rate):
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate

        
        return;        

    
    def build_model(self):

        input_position = Input(shape=(3,), name='cartesian_cor')
        x = Conv1D(filters=8,kernel_size=3,activation="relu")(input_position)

        input_lin_vel = Input(shape=(3,), name="linear_vel")
        x1 = Conv1D(filters=8,kernel_size=3,activation="relu")(input_lin_vel)

        input_rot_vel = Input(shape=(3,),name="rot_vel")
        x2 = Conv1D(filters=8,kernel_size=3,activation="relu")(input_rot_vel) 

        input_rot_mat = Input(shape=(9,), name="rot_mat")
        x3 = Conv1D(8,9,)


        pass