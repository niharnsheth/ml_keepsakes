import pandas as pd
import numpy as np
import os

class JigsawDataset():
    ''' Contains the jigsaw surgical dataset 
    '''
    _dataset_html  =  "https://drive.google.com/drive/folders/1z3Bj90KKA3bfQTqO62TvRwKNbyytWbrD?usp=share_link"
    _local_file_path  = r"D:/Documents/Nihar/JIGSAW_Dataset/jigsaw_dataset"
    _surgical_tasks = ["knot_tying", "needle_passing", "suturing"]


    def __init__(self, surgical_task):
        
        if surgical_task in self._surgical_tasks:
            print("task found !")
            self.surgical_task = surgical_task
        
        
        self.labels = self.load_labels(self.surgical_task)

        self.dataset_path = self._local_file_path + os.sep \
                            + self.surgical_task + os.sep \
                            + 'kinematics' + os.sep \
                            + 'AllGestures' + os.sep

        self.file_list = os.listdir(self.dataset_path)
        
        pass         


    def load_labels(self, surgical_task):
        meta_file_path = self._local_file_path + os.sep + \
                         self.surgical_task +  os.sep +  \
                         "meta_file_knot_tying.txt"
        
        column_names = ['filename', 'self_proclaimed_skill', 
                        'grs_total_score', 
                        'grs_respect_for_tissue_score',
                        'grs_inst_handling_score', 
                        'grs_time_motion_score',
                        'grs_flow_of_operation_score', 
                        'grs_overall_perf_score',
                        'grs_quality_of_final_product_score']

        labels_df = pd.read_csv(meta_file_path,
                                sep=r"\s+",
                                header=None,
                                names=column_names)
        
        return labels_df




