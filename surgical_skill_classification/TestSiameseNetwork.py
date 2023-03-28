## --- Libraries   ---  #
# File imports and aggregates data from multiple databases
import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


surgery_selected = 1 # for surgery_name_list and anchor_performance_name
# task_model_selected = 2

sliding_window = 120
# sliding_window_array = [[80, 345], [100, 90, 200, 140]]
# sliding_window_array = [[80, 80], [80, 80, 80, 80]]
#sliding_window_array = [[120, 120], [120, 120, 120, 120]]
sliding_window_array = [[140, 140], [140, 140, 140, 140]]
#step_size_for_windows = int(sliding_window)
step_size_for_windows = 40

number_of_labels = 1
n_features = 13

# File path to the database files
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
# source_path = os.getcwd() + "/../../../../Nihar/ML-data/SurgicalData"
source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData/Manually_Cleaned_And_Annotated_06272021'
anchor_performance_path = '/TrainingDataForComparison'
anchor_performance_name = ['1_Expert_412021_44525_PM.csv', '1_Expert_332021_30057_PM.csv']
# anchor_performance_name = ['1_Expert_412021_40051_PM.csv', '1_Expert_332021_12906_PM.csv']
test_data_path = '/TestData/TestingDataForComparison'
surgery_name_list = ['/Pericardiocentesis', '/Thoracentesis']

saved_to_folder = '/Results/Siamese'
# saved_model = '/07122021'
saved_model = '/08142021_flexSCNN_w140_ep200'

models_per_task = ['Pericardiocentesis_0', 'Pericardiocentesis_1',
                   'Thoracentesis_0', 'Thoracentesis_1', 'Thoracentesis_2', 'Thoracentesis_3']


##  ---   Make motion windows   --- #
# return an input and output data frame with motion windows
def create_motion_windows(window_span, df_to_change, step_size, number_of_features, number_of_labels):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - window_span)
    time_index = 0
    while time_index + window_span < len(df_to_change):
        a = df_to_change.iloc[time_index:time_index + window_span, :-number_of_labels].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[time_index + window_span, number_of_features:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
        time_index += step_size
    return local_feature_df, local_label_df


surgical_tasks_list = os.listdir(source_path + test_data_path + surgery_name_list[surgery_selected] + '/')
# get all the models saved for each task
# models_per_task = os.listdir(source_path + saved_to_folder[model_selected] + saved_model[model_selected])

# ## Predict values
# for surgical_task in surgical_tasks_list:
#
#     predicted_max_val = []
#     predicted_avg_val = []
#     # find the saved model to use
#     model_for_task = surgery_name_list[surgery_selected] + '_' + surgical_task
#     loaded_model = load_model(source_path + saved_to_folder +
#                               saved_model + '/' + model_for_task,
#                               custom_objects=None, compile=True)
#     print("Saved Model Path: " + source_path + saved_to_folder + saved_model + '/' + model_for_task)
#     # get the anchor performance to compare the testing data
#     anchor_performance_df = pd.read_csv(source_path + anchor_performance_path +
#                                         surgery_name_list[surgery_selected] + '/' +
#                                         str(surgical_task) + '/' +
#                                         anchor_performance_name[surgery_selected])
#
#     anchor_features, anchor_label = create_motion_windows(sliding_window,
#                                                           anchor_performance_df,
#                                                           step_size_for_windows,
#                                                           n_features,
#                                                           number_of_labels)
#     anchor_features = np.reshape(anchor_features, (len(anchor_features), sliding_window, n_features))
#     anchor_label = np.array(anchor_label)
#     # get the test data set file names
#     csv_list = [f for f in os.listdir(source_path + test_data_path +
#                                       surgery_name_list[surgery_selected] + '/' +
#                                       str(surgical_task) + '/')
#                 if fnmatch.fnmatch(f, '*.csv')]
#
#     header_list = []
#     # train for surgical task
#     for file in csv_list:
#         print(" --------    Current Surgical Task: " + surgical_task + "      ---------")
#         print(" --------    Predicting for file: " + file + "      ---------")
#         print(" --------    Using model: " + model_for_task + "      ---------")
#
#         max_val_predictions = []
#         avg_val_predictions = []
#
#         # read the file into a data-frame
#         df = pd.read_csv(source_path + test_data_path +
#                          surgery_name_list[surgery_selected] + '/' +
#                          str(surgical_task) + '/' + file)
#         header_list.append(file)
#         # create motion windows and separate data into input and output
#         # use the same step size as the size of the window
#         feature_list, label_list = create_motion_windows(sliding_window, df, step_size_for_windows, n_features, number_of_labels)
#         # create list of windows
#         feature_list = np.reshape(feature_list, (len(feature_list), sliding_window, n_features))
#         label_list = np.array(label_list)
#
#         # prediction = loaded_model.predict(anchor_features, feature_list)
#         # print(prediction * 100)
#         # print("----------------------------------------------------------------------------")
#         # final_feature_list_pairs, final_label_list_pairs = []
#         for i in range(len(feature_list)):
#             predictions = []
#             for j in range(len(anchor_features)):
#                 input_left = anchor_features[j]
#                 input_left = np.reshape(input_left, (1, sliding_window, n_features))
#                 input_right = feature_list[i]
#                 input_right = np.reshape(input_right, (1, sliding_window, n_features))
#                 prediction = loaded_model.predict([input_left, input_right])
#                 predictions.append(prediction[0,0])
#                 # print(prediction * 100)
#             print("Max values: " + str(np.amax(predictions)))
#             max_val_predictions.append(np.max(predictions))
#             print("Average value: " + str(np.average(predictions)))
#             avg_val_predictions.append(np.average(predictions))
#             print("----------------------------------------------------------------------------")
#     # max_val_df = pd.DataFrame(max_val_predictions)
#     # avg_val_df = pd.DataFrame(avg_val_predictions)
#     # max_val_np = np.array(max_val_predictions)
#     # avg_val_np = np.asarray(avg_val_predictions)
#         predicted_max_val.append(max_val_predictions)
#         predicted_avg_val.append(avg_val_predictions)
#
#     pd.DataFrame(predicted_max_val).T.to_csv(source_path + saved_to_folder +
#                                              saved_model + '/' + model_for_task +
#                                              '_prediction_max_value.csv', header=header_list)
#
#     pd.DataFrame(predicted_avg_val).T.to_csv(source_path + saved_to_folder +
#                                              saved_model + '/' + model_for_task +
#                                              '_prediction_avg_value.csv', header=header_list)
#


## Predict values
for surgical_task in range(len(surgical_tasks_list)):

    predicted_max_val = []
    predicted_avg_val = []
    # find the saved model to use
    model_for_task = surgery_name_list[surgery_selected] + '_' + surgical_tasks_list[surgical_task]
    loaded_model = load_model(source_path + saved_to_folder +
                              saved_model + '/' + model_for_task,
                              custom_objects=None, compile=True)
    print("Saved Model Path: " + source_path + saved_to_folder + saved_model + '/' + model_for_task)
    # get the anchor performance to compare the testing data
    anchor_performance_df = pd.read_csv(source_path + anchor_performance_path +
                                        surgery_name_list[surgery_selected] + '/' +
                                        str(surgical_tasks_list[surgical_task]) + '/' +
                                        anchor_performance_name[surgery_selected])

    anchor_features, anchor_label = create_motion_windows(sliding_window_array[surgery_selected][surgical_task],
                                                          anchor_performance_df,
                                                          step_size_for_windows,
                                                          n_features,
                                                          number_of_labels)
    anchor_features = np.reshape(anchor_features, (len(anchor_features),
                                                   sliding_window_array[surgery_selected][surgical_task],
                                                   n_features))
    anchor_label = np.array(anchor_label)
    # get the test data set file names
    csv_list = [f for f in os.listdir(source_path + test_data_path +
                                      surgery_name_list[surgery_selected] + '/' +
                                      str(surgical_tasks_list[surgical_task]) + '/')
                if fnmatch.fnmatch(f, '*.csv')]

    header_list = []
    # train for surgical task
    for file in csv_list:
        print(" --------    Current Surgical Task: " + surgical_tasks_list[surgical_task] + "      ---------")
        print(" --------    Predicting for file: " + file + "      ---------")
        print(" --------    Using model: " + model_for_task + "      ---------")

        max_val_predictions = []
        avg_val_predictions = []

        # read the file into a data-frame
        df = pd.read_csv(source_path + test_data_path +
                         surgery_name_list[surgery_selected] + '/' +
                         str(surgical_tasks_list[surgical_task]) + '/' + file)
        header_list.append(file)
        # create motion windows and separate data into input and output
        # use the same step size as the size of the window
        feature_list, label_list = create_motion_windows(sliding_window_array[surgery_selected][surgical_task], df, step_size_for_windows, n_features, number_of_labels)
        # create list of windows
        feature_list = np.reshape(feature_list, (len(feature_list),
                                                 sliding_window_array[surgery_selected][surgical_task],
                                                 n_features))
        label_list = np.array(label_list)

        # prediction = loaded_model.predict(anchor_features, feature_list)
        # print(prediction * 100)
        # print("----------------------------------------------------------------------------")
        # final_feature_list_pairs, final_label_list_pairs = []
        for i in range(len(feature_list)):
            predictions = []
            for j in range(len(anchor_features)):
                input_left = anchor_features[j]
                input_left = np.reshape(input_left, (1,
                                                     sliding_window_array[surgery_selected][surgical_task],
                                                     n_features))
                input_right = feature_list[i]
                input_right = np.reshape(input_right, (1,
                                                       sliding_window_array[surgery_selected][surgical_task],
                                                       n_features))
                prediction = loaded_model.predict([input_left, input_right])
                predictions.append(prediction[0,0])
                # print(prediction * 100)
            print("Max values: " + str(np.amax(predictions)))
            max_val_predictions.append(np.max(predictions))
            print("Average value: " + str(np.average(predictions)))
            avg_val_predictions.append(np.average(predictions))
            print("----------------------------------------------------------------------------")
    # max_val_df = pd.DataFrame(max_val_predictions)
    # avg_val_df = pd.DataFrame(avg_val_predictions)
    # max_val_np = np.array(max_val_predictions)
    # avg_val_np = np.asarray(avg_val_predictions)
        predicted_max_val.append(max_val_predictions)
        predicted_avg_val.append(avg_val_predictions)

        print("----- Final Prediction is ------ " )
        print("Max value:  " + str(np.amax(max_val_predictions)))
        print("Avg value:  " + str(np.average(max_val_predictions)))

    pd.DataFrame(predicted_max_val).T.to_csv(source_path + saved_to_folder +
                                             saved_model + '/' + model_for_task +
                                             '_prediction_max_value.csv', header=header_list)

    pd.DataFrame(predicted_avg_val).T.to_csv(source_path + saved_to_folder +
                                             saved_model + '/' + model_for_task +
                                             '_prediction_avg_value.csv', header=header_list)





##

