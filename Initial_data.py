import pandas as pd
import numpy as np
import torch


# 定义DataSet类
class DataSet():
    # 初始化函数，self指向的是实例化对象
    def __init__(self, basedir):
        self.basedir = basedir
        read_dir = basedir + '/model_input_data/'
        save_dir = basedir + '/model_output_data/'
        self.train_data = pd.read_csv(read_dir + "model_train_data_64.csv", index_col='username')
        self.test_data = pd.read_csv(read_dir + "model_test_data_64.csv", index_col='username')
        martix_user = pd.read_csv(read_dir + "avg_user_2.csv", index_col='course_id')
        martix_course = pd.read_csv(read_dir + "avg_course_2.csv", index_col='username')
        martix = pd.read_csv(read_dir + "adjacency_2.csv", index_col=0)

        self.martix_user = martix_user
        self.martix_course = martix_course
        self.martix = martix
        self.train_record = self.train_data
        self.test_record = self.test_data

        self.train_total_stu_list = list(set(self.train_record.index))
        self.test_total_stu_list = list(set(self.test_record.index))

        self.train_stu_num = len(set(self.train_data.index))
        self.train_cour_num = len(set(self.train_data['course_id']))

        self.test_stu_num = len(set(self.test_data.index))
        self.test_cour_num = len(set(self.test_data['course_id']))

        self.behavior_num = martix_user.shape[1]
        self.save_dir = save_dir

    def get_martix_user(self):
        martix_user = self.martix_user
        CA= np.array(martix_user.values.tolist())
        return torch.tensor(CA, dtype=torch.double)

    def get_martix_course(self):
        martix_course = self.martix_course
        UA = np.array(martix_course.values.tolist())
        return torch.tensor(UA, dtype=torch.double)

    def get_martix(self, ):
        martix = self.martix
        UC = np.array(martix.values.tolist())
        return torch.tensor(UC, dtype=torch.double)
