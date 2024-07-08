
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("../")
sys.path.append("./")
from utils.timefeatures import time_features
import ruptures as rpt
import warnings
import random

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='etth1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', time_change_detect=False, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_change_detect = time_change_detect

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def _getSTD(self, data: np.ndarray, windows_size: int):
        """
        arg:
            data: 等待计算方差的数据
            windows_size:   考虑计算方差的窗口大小
        return:
            std_data: 输出方差大小
        
        计算数据的方差
        """
        len = data.shape[0]
        std_data = np.zeros(len)
        before_windows_len = windows_size // 2
        after_windows_len = windows_size - before_windows_len - 1
        for i in range(before_windows_len):
            std_data[i] = data[0: before_windows_len + i + 1].std()
        for i in range(before_windows_len, len-after_windows_len):
            std_data[i] = data[i - before_windows_len: i + after_windows_len + 1].std()
        for i in range(len - after_windows_len, len):
            std_data[i] = data[i - before_windows_len: ].std()
        return std_data.reshape(-1, 1)
    


    def __time_change_detect__(self, data: np.array):
        std = StandardScaler()
        data_std = std.fit_transform(data)

        std_data_std = self._getSTD(data, self.pred_len+self.seq_len)
        std_data_std = std.fit_transform(std_data_std)
        data_std = np.concatenate([
                                data_std, 
                                std_data_std, 
                            ], axis=1)

        n = data_std.shape[0]
        dim = data_std.shape[1]
        sigma = np.std(data_std)

        algo = rpt.Pelt(model="l2", min_size=self.seq_len+self.pred_len).fit(data_std)
        my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
        return my_bkps
    
    def __add_bkps_to_data__(self, data: np.array, bkps: list):
        """
        data: [features, ..., bkps]
        """
        if not self.set_type:
            _label = 0
            cpd_label = np.zeros((data.shape[0], 1))
            start_index = 0
            for i in bkps:
                end_index = i
                cpd_label[start_index: end_index, 0] = _label
                _label = 1 - _label
                start_index = end_index
        else:
            cpd_label = np.ones((data.shape[0], 1))
        data = np.concatenate([data, cpd_label], axis = 1)
        return data




    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0,            12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24,  12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        

        # 标记是单变量还是双变量 

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        if self.time_change_detect:
            if not self.set_type:
                bkps = self.__time_change_detect__(data)
            else:
                bkps = None
            data = self.__add_bkps_to_data__(data, bkps)
        

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __process_cpd_label__(self, cpd_label):
        if cpd_label[-1][0] == 0:
            cpd_label = 1 - cpd_label
        else:
            pass
        return cpd_label

    def __getitem__(self, index):


        s_begin = index
        s_end = s_begin + self.seq_len
        
        # x y 有一定的重复
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.time_change_detect:
            seq_x = self.data_x[s_begin:s_end, :-1]
            seq_y = self.data_y[r_begin:r_end, :-1]
            cpd_label = self.__process_cpd_label__(self.data_x[s_begin:s_end, -1:])
            seq_x = np.concatenate([seq_x, cpd_label], axis = 1)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.data_x.shape[0] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ettm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', time_change_detect=False, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_change_detect = time_change_detect

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def _getSTD(self, data: np.ndarray, windows_size: int):
        """
        arg:
            data: 等待计算方差的数据
            windows_size:   考虑计算方差的窗口大小
        return:
            std_data: 输出方差大小
        
        计算数据的方差
        """
        len = data.shape[0]
        std_data = np.zeros(len)
        before_windows_len = windows_size // 2
        after_windows_len = windows_size - before_windows_len - 1
        for i in range(before_windows_len):
            std_data[i] = data[0: before_windows_len + i + 1].std()
        for i in range(before_windows_len, len-after_windows_len):
            std_data[i] = data[i - before_windows_len: i + after_windows_len + 1].std()
        for i in range(len - after_windows_len, len):
            std_data[i] = data[i - before_windows_len: ].std()
        return std_data.reshape(-1, 1)
    


    def __time_change_detect__(self, data: np.array):
        std = StandardScaler()
        data_std = std.fit_transform(data)

        std_data_std = self._getSTD(data, self.pred_len+self.seq_len)
        std_data_std = std.fit_transform(std_data_std)
        data_std = np.concatenate([
                                data_std, 
                                std_data_std, 
                            ], axis=1)

        n = data_std.shape[0]
        dim = data_std.shape[1]
        sigma = np.std(data_std)

        algo = rpt.Pelt(model="l2", min_size=self.seq_len+self.pred_len).fit(data_std)
        my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
        return my_bkps

    def __add_bkps_to_data__(self, data: np.array, bkps: list):
        """
        data: [features, ..., bkps]
        """
        if not self.set_type:
            _label = 0
            cpd_label = np.zeros((data.shape[0], 1))
            start_index = 0
            for i in bkps:
                end_index = i
                cpd_label[start_index: end_index, 0] = _label
                _label = 1 - _label
                start_index = end_index
        else:
            cpd_label = np.ones((data.shape[0], 1))
        data = np.concatenate([data, cpd_label], axis = 1)
        return data

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        if self.time_change_detect:
            if not self.set_type:
                bkps = self.__time_change_detect__(data)
            else:
                bkps = None
            data = self.__add_bkps_to_data__(data, bkps)


        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __process_cpd_label__(self, cpd_label):
        if cpd_label[-1][0] == 0:
            cpd_label = 1 - cpd_label
        else:
            pass
        return cpd_label

    def __getitem__(self, index):


        s_begin = index
        s_end = s_begin + self.seq_len
        
        # x y 有一定的重复
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.time_change_detect:
            seq_x = self.data_x[s_begin:s_end, :-1]
            seq_y = self.data_y[r_begin:r_end, :-1]
            cpd_label = self.__process_cpd_label__(self.data_x[s_begin:s_end, -1:])
            seq_x = np.concatenate([seq_x, cpd_label], axis = 1)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.data_x.shape[0] - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_EXCHANGE(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='exchange.csv',
                 target='0', scale=True, timeenc=0, freq='h', time_change_detect=False, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_change_detect = time_change_detect

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def _getSTD(self, data: np.ndarray, windows_size: int):

        len = data.shape[0]
        std_data = np.zeros(len)
        before_windows_len = windows_size // 2
        after_windows_len = windows_size - before_windows_len - 1
        for i in range(before_windows_len):
            std_data[i] = data[0: before_windows_len + i + 1].std()
        for i in range(before_windows_len, len-after_windows_len):
            std_data[i] = data[i - before_windows_len: i + after_windows_len + 1].std()
        for i in range(len - after_windows_len, len):
            std_data[i] = data[i - before_windows_len: ].std()
        return std_data.reshape(-1, 1)
    


    def __time_change_detect__(self, data: np.array):
        std = StandardScaler()
        data_std = std.fit_transform(data)

        std_data_std = self._getSTD(data, self.pred_len+self.seq_len)
        std_data_std = std.fit_transform(std_data_std)
        data_std = np.concatenate([
                                data_std, 
                                std_data_std, 
                            ], axis=1)
        n = data_std.shape[0]
        dim = data_std.shape[1]
        sigma = np.std(data_std)

        algo = rpt.Pelt(model="l2", min_size=self.seq_len+self.pred_len).fit(data_std)
        my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
        return my_bkps
    
    def __add_bkps_to_data__(self, data: np.array, bkps: list):

        if not self.set_type:
            _label = 0
            cpd_label = np.zeros((data.shape[0], 1))
            start_index = 0
            for i in bkps:
                end_index = i
                cpd_label[start_index: end_index, 0] = _label
                _label = 1 - _label
                start_index = end_index
        else:
            cpd_label = np.ones((data.shape[0], 1))
        data = np.concatenate([data, cpd_label], axis = 1)
        return data




    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # randseed = 0
        # random.seed(randseed)
        # index_list = [i for i in range(len(df_raw) - self.seq_len - self.pred_len + 1)]
        # random.shuffle(index_list)
        
        border1s = [0,            12 * 30 * 12 - self.seq_len, 12 * 30 * 12 + 12 * 30 * 4 - self.seq_len]
        border2s = [12 * 30 * 12, 12 * 30 * 12 + 12 * 30 * 4,  12 * 30 * 16 + 12 * 30 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 标记是单变量还是双变量 

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        if self.time_change_detect:
            if not self.set_type:
                bkps = self.__time_change_detect__(data)
            else:
                bkps = None
            data = self.__add_bkps_to_data__(data, bkps)

        

        df_stamp = df_raw[['Unnamed: 0']].values[border1:border2]
        # print(df_stamp.shape)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = np.ones_like(df_stamp)



        # self.data = data
        # self.index = index_list[border1: border2]
        # self.len = border2 - border1
        # df_stamp = df_raw[['Unnamed: 0']].values
        # self.data_stamp = df_stamp
    
    def __process_cpd_label__(self, cpd_label):
        if cpd_label[-1][0] == 0:
            cpd_label = 1 - cpd_label
        else:
            pass
        return cpd_label

    def __getitem__(self, index):

        s_begin = index
        # s_begin = self.index[index]
        s_end = s_begin + self.seq_len
        
        # x y 有一定的重复
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # if self.time_change_detect:
        #     seq_x = self.data[s_begin:s_end, :-1]
        #     seq_y = self.data[r_begin:r_end, :-1]
        #     cpd_label = self.__process_cpd_label__(self.data[s_begin:s_end, -1:])
        #     seq_x = np.concatenate([seq_x, cpd_label], axis = 1)
        # else:
        #     seq_x = self.data[s_begin:s_end]
        #     seq_y = self.data[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.time_change_detect:
            seq_x = self.data_x[s_begin:s_end, :-1]
            seq_y = self.data_y[r_begin:r_end, :-1]
            cpd_label = self.__process_cpd_label__(self.data_x[s_begin:s_end, -1:])
            seq_x = np.concatenate([seq_x, cpd_label], axis = 1)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # return self.len - self.seq_len - self.pred_len + 1
        return self.data_x.shape[0] - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class Dataset_SOLAR(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='solar.csv',
                 target='Power(MW)', scale=True, timeenc=0, freq='t', time_change_detect=False, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_change_detect = time_change_detect

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def _getSTD(self, data: np.ndarray, windows_size: int):
        """
        arg:
            data: 等待计算方差的数据
            windows_size:   考虑计算方差的窗口大小
        return:
            std_data: 输出方差大小
        
        计算数据的方差
        """
        len = data.shape[0]
        std_data = np.zeros(len)
        before_windows_len = windows_size // 2
        after_windows_len = windows_size - before_windows_len - 1
        for i in range(before_windows_len):
            std_data[i] = data[0: before_windows_len + i + 1].std()
        for i in range(before_windows_len, len-after_windows_len):
            std_data[i] = data[i - before_windows_len: i + after_windows_len + 1].std()
        for i in range(len - after_windows_len, len):
            std_data[i] = data[i - before_windows_len: ].std()
        return std_data.reshape(-1, 1)
    


    def __time_change_detect__(self, data: np.array):
        std = StandardScaler()
        data_std = std.fit_transform(data)

        std_data_std = self._getSTD(data, self.pred_len+self.seq_len)
        std_data_std = std.fit_transform(std_data_std)
        data_std = np.concatenate([
                                data_std, 
                                std_data_std, 
                            ], axis=1)

        n = data_std.shape[0]
        dim = data_std.shape[1]
        sigma = np.std(data_std)

        algo = rpt.Pelt(model="l2", min_size=self.seq_len+self.pred_len).fit(data_std)
        my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
        return my_bkps

    def __add_bkps_to_data__(self, data: np.array, bkps: list):
        """
        data: [features, ..., bkps]
        """
        if not self.set_type:
            _label = 0
            cpd_label = np.zeros((data.shape[0], 1))
            start_index = 0
            for i in bkps:
                end_index = i
                cpd_label[start_index: end_index, 0] = _label
                _label = 1 - _label
                start_index = end_index
        else:
            cpd_label = np.ones((data.shape[0], 1))
        data = np.concatenate([data, cpd_label], axis = 1)
        return data

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw.replace("?", np.NaN, inplace=True)
        border1s = [0, 12 * 12 * 24 * 18 - self.seq_len, 12 * 12 * 24 * 24 - self.seq_len]
        border2s = [12 * 12 * 24 * 18, 12 * 12 * 24 * 24, 12 * 12 * 24 * 30]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data].astype('float32').interpolate()
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype('float32').interpolate()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        if self.time_change_detect:
            if not self.set_type:
                bkps = self.__time_change_detect__(data)
            else:
                bkps = None
            data = self.__add_bkps_to_data__(data, bkps)


        df_stamp = df_raw[['LocalTime']][border1:border2]
        df_stamp['LocalTime'] = pd.to_datetime(df_stamp.loc[:, 'LocalTime'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['LocalTime'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['LocalTime'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __process_cpd_label__(self, cpd_label):
        if cpd_label[-1][0] == 0:
            cpd_label = 1 - cpd_label
        else:
            pass
        return cpd_label

    def __getitem__(self, index):


        s_begin = index
        s_end = s_begin + self.seq_len
        
        # x y 有一定的重复
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.time_change_detect:
            seq_x = self.data_x[s_begin:s_end, :-1]
            seq_y = self.data_y[r_begin:r_end, :-1]
            cpd_label = self.__process_cpd_label__(self.data_x[s_begin:s_end, -1:])
            seq_x = np.concatenate([seq_x, cpd_label], axis = 1)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.data_x.shape[0] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
        
class Dataset_SUMSTEPS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='sumsteps.csv',
                 target='steps', scale=True, timeenc=0, freq='t', time_change_detect=False, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_change_detect = time_change_detect

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def _getSTD(self, data: np.ndarray, windows_size: int):
        """
        arg:
            data: 等待计算方差的数据
            windows_size:   考虑计算方差的窗口大小
        return:
            std_data: 输出方差大小
        
        计算数据的方差
        """
        len = data.shape[0]
        std_data = np.zeros(len)
        before_windows_len = windows_size // 2
        after_windows_len = windows_size - before_windows_len - 1
        for i in range(before_windows_len):
            std_data[i] = data[0: before_windows_len + i + 1].std()
        for i in range(before_windows_len, len-after_windows_len):
            std_data[i] = data[i - before_windows_len: i + after_windows_len + 1].std()
        for i in range(len - after_windows_len, len):
            std_data[i] = data[i - before_windows_len: ].std()
        return std_data.reshape(-1, 1)
    


    def __time_change_detect__(self, data: np.array):
        std = StandardScaler()
        data_std = std.fit_transform(data)

        std_data_std = self._getSTD(data, self.pred_len+self.seq_len)
        std_data_std = std.fit_transform(std_data_std)
        data_std = np.concatenate([
                                data_std, 
                                std_data_std, 
                            ], axis=1)

        n = data_std.shape[0]
        dim = data_std.shape[1]
        sigma = np.std(data_std)

        algo = rpt.Pelt(model="l2", min_size=self.seq_len+self.pred_len).fit(data_std)
        my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
        return my_bkps

    def __add_bkps_to_data__(self, data: np.array, bkps: list):
        """
        data: [features, ..., bkps]
        """
        if not self.set_type:
            _label = 0
            cpd_label = np.zeros((data.shape[0], 1))
            start_index = 0
            for i in bkps:
                end_index = i
                cpd_label[start_index: end_index, 0] = _label
                _label = 1 - _label
                start_index = end_index
        else:
            cpd_label = np.ones((data.shape[0], 1))
        data = np.concatenate([data, cpd_label], axis = 1)
        return data

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path)).iloc[:-1, :]
        df_raw.replace("?", np.NaN, inplace=True)
        border1s = [0, 144 * 210 - self.seq_len, 144 * 280 - self.seq_len]
        border2s = [144 * 210, 144 * 280, 144 * 350]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data].astype('float32').interpolate()
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype('float32').interpolate()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        if self.time_change_detect:
            if not self.set_type:
                bkps = self.__time_change_detect__(data)
            else:
                bkps = None
            data = self.__add_bkps_to_data__(data, bkps)


        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.loc[:, 'date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __process_cpd_label__(self, cpd_label):
        if cpd_label[-1][0] == 0:
            cpd_label = 1 - cpd_label
        else:
            pass
        return cpd_label

    def __getitem__(self, index):


        s_begin = index
        s_end = s_begin + self.seq_len
        
        # x y 有一定的重复
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.time_change_detect:
            seq_x = self.data_x[s_begin:s_end, :-1]
            seq_y = self.data_y[r_begin:r_end, :-1]
            cpd_label = self.__process_cpd_label__(self.data_x[s_begin:s_end, -1:])
            seq_x = np.concatenate([seq_x, cpd_label], axis = 1)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # print(type(seq_x), type(seq_y), type(seq_x_mark), type(seq_y_mark))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.data_x.shape[0] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', time_change_detect=False, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ARTIFICIAL(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='artificial.csv',
                 target='mean', scale=True, timeenc=0, freq='t', time_change_detect=False, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_change_detect = time_change_detect

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def _getSTD(self, data: np.ndarray, windows_size: int):
        """
        arg:
            data: 等待计算方差的数据
            windows_size:   考虑计算方差的窗口大小
        return:
            std_data: 输出方差大小
        
        计算数据的方差
        """
        len = data.shape[0]
        std_data = np.zeros(len)
        before_windows_len = windows_size // 2
        after_windows_len = windows_size - before_windows_len - 1
        for i in range(before_windows_len):
            std_data[i] = data[0: before_windows_len + i + 1].std()
        for i in range(before_windows_len, len-after_windows_len):
            std_data[i] = data[i - before_windows_len: i + after_windows_len + 1].std()
        for i in range(len - after_windows_len, len):
            std_data[i] = data[i - before_windows_len: ].std()
        return std_data.reshape(-1, 1)
    


    def __time_change_detect__(self, data: np.array):
        std = StandardScaler()
        data_std = std.fit_transform(data)

        std_data_std = self._getSTD(data, self.pred_len+self.seq_len)
        std_data_std = std.fit_transform(std_data_std)
        data_std = np.concatenate([
                                data_std, 
                                std_data_std, 
                            ], axis=1)

        n = data_std.shape[0]
        dim = data_std.shape[1]
        sigma = np.std(data_std)

        algo = rpt.Binseg(model="l2", min_size=self.seq_len+self.pred_len, jump=5).fit(data_std)
        my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
        return my_bkps

    # def __add_bkps_to_data__(self, data: np.array, bkps: list):
    #     """
    #     data: [features, ..., bkps]
    #     """
    #     if not self.set_type:
    #         _label = 0
    #         cpd_label = np.zeros((data.shape[0], 1))
    #         start_index = 0
    #         for i in bkps:
    #             end_index = i
    #             cpd_label[start_index: end_index, 0] = _label
    #             _label = 1 - _label
    #             start_index = end_index
    #     else:
    #         cpd_label = np.ones((data.shape[0], 1))

    def __add_bkps_to_data__(self, data: np.array, bkps: list):
        """
        data: [features, ..., bkps]
        """
        # if not self.set_type:
        #     _label = 0
        #     cpd_label = np.zeros((data.shape[0], 1))
        #     start_index = 0
        #     for i in bkps:
        #         end_index = i
        #         cpd_label[start_index: end_index, 0] = _label
        #         _label = 1 - _label
        #         start_index = end_index
        # else:
        #     cpd_label = np.ones((data.shape[0], 1))

        _label = 0
        cpd_label = np.zeros((data.shape[0], 1))
        start_index = 0
        for i in bkps:
            end_index = i
            cpd_label[start_index: end_index, 0] = _label
            _label = 1 - _label
            start_index = end_index
        
        data = np.concatenate([data, cpd_label], axis = 1)
        return data

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        border1s = [0, int(6*1e4 - self.seq_len), int(8*1e4 - self.seq_len)]
        border2s = [int(6*1e4), int(8*1e4), int(1e5)]
        border1 = int(border1s[self.set_type])
        border2 = int(border2s[self.set_type])

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data].astype('float32').interpolate()
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype('float32').interpolate()

        if self.scale:
            train_data = df_data[int(border1s[0]): int(border2s[0])]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        if self.time_change_detect:
            # if not self.set_type:
            #     bkps = self.__time_change_detect__(data)
            # else:
            #     bkps = None

            # # 手动检测 
            # bkps = self.__time_change_detect__(data)
            # data = self.__add_bkps_to_data__(data, bkps)

            # 不进行检测
            _mask_data = df_raw[[self.target+"_mask"]].astype('float32').interpolate().values
            data = np.concatenate([data, _mask_data], axis=1)

        df_stamp = df_raw[['Unnamed: 0']].values[border1:border2]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = df_stamp
    
    def __process_cpd_label__(self, cpd_label):
        if cpd_label[-1][0] == 0:
            cpd_label = 1 - cpd_label
        else:
            pass
        return cpd_label

    def __getitem__(self, index):


        s_begin = index
        s_end = s_begin + self.seq_len
        
        # x y 有一定的重复
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.time_change_detect:
            seq_x = self.data_x[s_begin:s_end, :-1]
            seq_y = self.data_y[r_begin:r_end, :-1]
            cpd_label = self.__process_cpd_label__(self.data_x[s_begin:s_end, -1:])
            seq_x = np.concatenate([seq_x, cpd_label], axis = 1)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.data_x.shape[0] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', time_change_detect=False, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', time_change_detect=False, cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        

        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



if __name__ == "__main__":
    dataset = Dataset_ETT_hour(root_path = "../dataset/", flag='train', data_path = "etth1.csv", time_change_detect=True)
    print(len(dataset))
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
    print(seq_x.shape)
    print(seq_y.shape)

