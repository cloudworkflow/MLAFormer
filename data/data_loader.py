import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn.preprocessing

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_ts2vec = sklearn.preprocessing.StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # df_raw=pd.read_csv(os.path.join(self.root_path,  self.data_path),index_col='date', parse_dates=True)
        # dt_embed=_get_time_features(df_raw.index)

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # dt_embed=df_raw.index
        # dt_embed=time_features(dt_embed, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp =data_stamp[border1:border2]

        dt_embed=_get_time_features(df_stamp['date'].dt)
        self.scaler_ts2vec.fit(dt_embed[border1s[0]:border2s[0]])
        dt_embed=self.scaler_ts2vec.transform(dt_embed)[border1:border2]
        self.data_ts2vec=np.concatenate([dt_embed, self.data_x], axis=-1)

    def __getitem__(self, index):
        s_begin = index  #0
        s_end = s_begin + self.seq_len  #96
        r_begin = s_end - self.label_len  #48
        r_end = r_begin + self.label_len + self.pred_len  #96+24

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]   #0- 96
        seq_y_mark = self.data_stamp[r_begin:r_end]   #48- 96+24

        return seq_x, seq_y, seq_x_mark, seq_y_mark#,self.data_ts2vec[s_begin:s_end]
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour_missing(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h',mr=0.1, cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq
        self.mr=mr
        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        self.scaler2=StandardScaler()
        df_raw=pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
        # df_raw=pd.read_csv(os.path.join(self.root_path,  self.data_path),index_col='date', parse_dates=True)
        # dt_embed=_get_time_features(df_raw.index)
        df_missing=df_raw.copy()
        idx=list(range(len(df_missing)))
        cols=df_missing.columns[1:]
        for i in range(len(cols)):
            random.seed(i*10)
            s=random.sample(idx, int(len(df_missing)*self.mr))
            s.sort()
            df_missing.loc[s, cols[i]]=np.nan
        for column in list(df_missing.columns[df_missing.isnull().sum()>0]):
            mean_val=df_missing[column].mean()
            df_missing[column].fillna(mean_val, inplace=True)

        border1s=[0, 12*30*24-self.seq_len, 12*30*24+4*30*24-self.seq_len]
        border2s=[12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=cols
            df_data_missing=df_missing[cols_data]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data_missing=df_missing[[self.target]]
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data_missing=df_data_missing[border1s[0]:border2s[0]]
            self.scaler.fit(train_data_missing.values)
            data_missing=self.scaler.transform(df_data_missing.values)
            train_data=df_data[border1s[0]:border2s[0]]
            data=self.scaler.transform(df_data.values)
        else:
            data_missing=df_data_missing.values
            data=df_data.values

        df_stamp=df_raw[['date']]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # dt_embed=df_raw.index
        # dt_embed=time_features(dt_embed, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data_missing[border1:border2]
        self.data_x_true=data[border1:border2]

        if self.inverse:
            self.data_y=df_data_missing.values[border1:border2]
            self.data_y_true=df_data.values[border1:border2]
        else:
            self.data_y=data_missing[border1:border2]
            self.data_y_true=data[border1:border2]
        self.data_stamp=data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin=index  # 0
        s_end=s_begin+self.seq_len  # 96
        r_begin=s_end-self.label_len  # 48
        r_end=r_begin+self.label_len+self.pred_len  # 96+24

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
            seq_y_true=np.concatenate([self.data_x_true[r_begin:r_begin+self.label_len], self.data_y_true[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
            seq_y_true=self.data_y_true[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]  # 0- 96
        seq_y_mark=self.data_stamp[r_begin:r_end]  # 48- 96+24

        return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_y_true  # ,self.data_ts2vec[s_begin:s_end]

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_hour_vlength(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq

        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        self.scaler_ts2vec=sklearn.preprocessing.StandardScaler()
        df_raw=pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
        # df_raw=pd.read_csv(os.path.join(self.root_path,  self.data_path),index_col='date', parse_dates=True)
        # dt_embed=_get_time_features(df_raw.index)

        border1s=[0, 12*30*24-self.seq_len, 12*30*24+4*30*24-self.seq_len]
        border2s=[12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=df_raw.columns[1:]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data=df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data=self.scaler.transform(df_data.values)
        else:
            data=df_data.values

        df_stamp=df_raw[['date']]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # dt_embed=df_raw.index
        # dt_embed=time_features(dt_embed, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data[border1:border2]
        if self.inverse:
            self.data_y=df_data.values[border1:border2]
        else:
            self.data_y=data[border1:border2]
        self.data_stamp=data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin=index  # 0
        random.seed(index*10)
        seq_pre=random.randint(12,self.seq_len)
        s_end = s_begin + self.seq_len  #96
        r_begin = s_end - self.label_len  #48
        r_end = r_begin + self.label_len + self.pred_len  #96+24

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]   #0- 96
        seq_y_mark = self.data_stamp[r_begin:r_end]   #48- 96+24
        seq_x  = seq_x[:,seq_pre:,]
        return seq_x, seq_y, seq_x_mark, seq_y_mark#,self.data_ts2vec[s_begin:s_end]


    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour_deepar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq

        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        self.scaler_ts2vec=sklearn.preprocessing.StandardScaler()
        df_raw=pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
        # df_raw=pd.read_csv(os.path.join(self.root_path,  self.data_path),index_col='date', parse_dates=True)
        # dt_embed=_get_time_features(df_raw.index)

        border1s=[0, 12*30*24-self.seq_len, 12*30*24+4*30*24-self.seq_len]
        border2s=[12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=df_raw.columns[1:]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data=df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data=self.scaler.transform(df_data.values)
        else:
            data=df_data.values

        df_stamp=df_raw[['date']]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # dt_embed=df_raw.index
        # dt_embed=time_features(dt_embed, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data[border1:border2]
        if self.inverse:
            self.data_y=df_data.values[border1:border2]
        else:
            self.data_y=data[border1:border2]
        self.data_stamp=data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin=index  # 0
        s_end=s_begin+self.seq_len  # 96
        r_begin=s_end-self.label_len  # 48
        r_end=r_begin+self.label_len+self.pred_len  # 96+24

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]  # 0- 96
        seq_y_mark=self.data_stamp[r_begin:r_end]  # 48- 96+24

        return seq_y, seq_y_mark  # ,self.data_ts2vec[s_begin:s_end]

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_ts2vec = sklearn.preprocessing.StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # df_raw=pd.read_csv(os.path.join(self.root_path,  self.data_path),index_col='date', parse_dates=True)
        # dt_embed=_get_time_features(df_raw.index)

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # dt_embed=df_raw.index
        # dt_embed=time_features(dt_embed, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp =data_stamp[border1:border2]

        dt_embed=_get_time_features(df_stamp['date'].dt)
        self.scaler_ts2vec.fit(dt_embed[border1s[0]:border2s[0]])
        dt_embed=self.scaler_ts2vec.transform(dt_embed)[border1:border2]
        self.data_ts2vec=np.concatenate([dt_embed, self.data_x], axis=-1)
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark #,self.data_ts2vec[s_begin:s_end]
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute_missing(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', mr=0.1,cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq
        self.mr=mr
        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        self.scaler2=StandardScaler()
        df_raw=pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
        # df_raw=pd.read_csv(os.path.join(self.root_path,  self.data_path),index_col='date', parse_dates=True)
        # dt_embed=_get_time_features(df_raw.index)
        df_missing=df_raw.copy()
        idx=list(range(len(df_missing)))
        cols=df_missing.columns[1:]
        for i in range(len(cols)):
            random.seed(i*10)
            s=random.sample(idx, int(len(df_missing)*self.mr))
            s.sort()
            df_missing.loc[s, cols[i]]=0
        border1s=[0, 12*30*24*4-self.seq_len, 12*30*24*4+4*30*24*4-self.seq_len]
        border2s=[12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=cols
            df_data_missing=df_missing[cols_data]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data_missing=df_missing[[self.target]]
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data_missing=df_data_missing[border1s[0]:border2s[0]]
            self.scaler.fit(train_data_missing.values)
            data_missing=self.scaler.transform(df_data_missing.values)
            train_data=df_data[border1s[0]:border2s[0]]
            data=self.scaler.transform(df_data.values)
        else:
            data_missing=df_data_missing.values
            data=df_data.values

        df_stamp=df_raw[['date']]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # dt_embed=df_raw.index
        # dt_embed=time_features(dt_embed, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data_missing[border1:border2]
        self.data_x_true=data[border1:border2]

        if self.inverse:
            self.data_y=df_data_missing.values[border1:border2]
            self.data_y_true=df_data.values[border1:border2]
        else:
            self.data_y=data_missing[border1:border2]
            self.data_y_true=data[border1:border2]
        self.data_stamp=data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin=index
        s_end=s_begin+self.seq_len
        r_begin=s_end-self.label_len
        r_end=r_begin+self.label_len+self.pred_len

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
            seq_y_true=np.concatenate(
                [self.data_x_true[r_begin:r_begin+self.label_len], self.data_y_true[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
            seq_y_true=self.data_y_true[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]
        seq_y_mark=self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_true

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute_deepar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq

        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        self.scaler_ts2vec=sklearn.preprocessing.StandardScaler()
        df_raw=pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
        # df_raw=pd.read_csv(os.path.join(self.root_path,  self.data_path),index_col='date', parse_dates=True)
        # dt_embed=_get_time_features(df_raw.index)

        border1s=[0, 12*30*24*4-self.seq_len, 12*30*24*4+4*30*24*4-self.seq_len]
        border2s=[12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=df_raw.columns[1:]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data=df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data=self.scaler.transform(df_data.values)
        else:
            data=df_data.values

        df_stamp=df_raw[['date']]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # dt_embed=df_raw.index
        # dt_embed=time_features(dt_embed, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data[border1:border2]
        if self.inverse:
            self.data_y=df_data.values[border1:border2]
        else:
            self.data_y=data[border1:border2]

        # self.scaler_ts2vec.fit(data_stamp[border1s[0]:border2s[0]])
        # dt_embed=self.scaler_ts2vec.transform(data_stamp)[border1:border2]
        # self.data_ts2vec=np.concatenate([dt_embed, self.data_x], axis=-1)
        # dt_embed=_get_time_features(df_stamp['date'].dt)
        # self.scaler_ts2vec.fit(dt_embed[border1s[0]:border2s[0]])
        # dt_embed=self.scaler_ts2vec.transform(dt_embed)[border1:border2]
        # self.data_ts2vec=np.concatenate([dt_embed, self.data_x], axis=-1)
        #
        # data_stamp=data_stamp[border1:border2]
        # self.data_stamp = data_stamp
        self.data_stamp=data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin=index
        s_end=s_begin+self.seq_len
        r_begin=s_end-self.label_len
        r_end=r_begin+self.label_len+self.pred_len

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]
        seq_y_mark=self.data_stamp[r_begin:r_end]

        return seq_y, seq_y_mark  # ,self.data_ts2vec[s_begin:s_end]

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_bus(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=True, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq
        self.cols=cols
        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        df_raw=pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols=list(df_raw.columns);
            cols.remove(self.target);
            cols.remove('date')
        df_raw=df_raw[['date']+cols+[self.target]]

        num_train=int(len(df_raw)*0.7)  # 70
        num_test=int(len(df_raw)*0.2)  # 20
        num_vali=len(df_raw)-num_train-num_test  # 10
        border1s=[0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]  # [0,65,100-20-5=75]
        border2s=[num_train, num_train+num_vali, len(df_raw)]  # [70,80,100]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=df_raw.columns[1:]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data=df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data=self.scaler.transform(df_data.values)
        else:
            data=df_data.values

        df_stamp=df_raw[['date']][border1:border2]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data[border1:border2]
        if self.inverse:
            self.data_y=df_data.values[border1:border2]
        else:
            self.data_y=data[border1:border2]
        self.data_stamp=data_stamp

    def __getitem__(self, index):
        s_begin=index
        s_end=s_begin+self.seq_len
        r_begin=s_end-self.label_len
        r_end=r_begin+self.label_len+self.pred_len

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]
        seq_y_mark=self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq
        self.cols=cols
        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        self.scaler_ts2vec=sklearn.preprocessing.StandardScaler()
        df_raw=pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols=list(df_raw.columns);
            cols.remove(self.target);
            cols.remove('date')
        df_raw=df_raw[['date']+cols+[self.target]]

        num_train=int(len(df_raw)*0.7)  # 70
        num_test=int(len(df_raw)*0.2)  # 20
        num_vali=len(df_raw)-num_train-num_test  # 10
        border1s=[0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]  # [0,65,100-20-5=75]
        border2s=[num_train, num_train+num_vali, len(df_raw)]  # [70,80,100]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=df_raw.columns[1:]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data=df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data=self.scaler.transform(df_data.values)
        else:
            data=df_data.values

        df_stamp=df_raw[['date']]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data[border1:border2]
        if self.inverse:
            self.data_y=df_data.values[border1:border2]
        else:
            self.data_y=data[border1:border2]
        self.data_stamp=data_stamp

        dt_embed=_get_time_features(df_stamp['date'].dt)
        self.scaler_ts2vec.fit(dt_embed[border1s[0]:border2s[0]])
        dt_embed=self.scaler_ts2vec.transform(dt_embed)[border1:border2]
        self.data_ts2vec=np.concatenate([dt_embed, self.data_x], axis=-1)
    def __getitem__(self, index):
        s_begin=index
        s_end=s_begin+self.seq_len
        r_begin=s_end-self.label_len
        r_end=r_begin+self.label_len+self.pred_len

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]
        seq_y_mark=self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark#, self.data_ts2vec[s_begin:s_end]

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_missing(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h',mr=0.1, cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq
        self.mr=mr
        self.cols=cols
        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        df_raw=pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
        df_missing=df_raw.copy()
        idx=list(range(len(df_missing)))
        cols=df_missing.columns[1:]
        for i in range(len(cols)):
            random.seed(i*10)
            s=random.sample(idx, int(len(df_missing)*self.mr))
            s.sort()
            df_missing.loc[s, cols[i]]=0
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols=list(df_raw.columns);
            cols.remove(self.target);
            cols.remove('date')
        df_raw=df_raw[['date']+cols+[self.target]]

        num_train=int(len(df_raw)*0.7)  # 70
        num_test=int(len(df_raw)*0.2)  # 20
        num_vali=len(df_raw)-num_train-num_test  # 10
        border1s=[0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]  # [0,65,100-20-5=75]
        border2s=[num_train, num_train+num_vali, len(df_raw)]  # [70,80,100]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=df_raw.columns[1:]
            df_data_missing=df_missing[cols_data]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data_missing=df_missing[[self.target]]
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data_missing=df_data_missing[border1s[0]:border2s[0]]
            self.scaler.fit(train_data_missing.values)
            data_missing=self.scaler.transform(df_data_missing.values)
            train_data=df_data[border1s[0]:border2s[0]]
            data=self.scaler.transform(df_data.values)
        else:
            data_missing=df_data_missing.values
            data=df_data.values

        df_stamp=df_raw[['date']][border1:border2]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data_missing[border1:border2]
        self.data_x_true=data[border1:border2]

        if self.inverse:
            self.data_y=df_data_missing.values[border1:border2]
            self.data_y_true=df_data.values[border1:border2]
        else:
            self.data_y=data_missing[border1:border2]
            self.data_y_true=data[border1:border2]
        self.data_stamp=data_stamp

    def __getitem__(self, index):
        s_begin=index
        s_end=s_begin+self.seq_len
        r_begin=s_end-self.label_len
        r_end=r_begin+self.label_len+self.pred_len

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
            seq_y_true=np.concatenate(
                [self.data_x_true[r_begin:r_begin+self.label_len], self.data_y_true[r_begin+self.label_len:r_end]], 0)

        else:
            seq_y=self.data_y[r_begin:r_end]
            seq_y_true=self.data_y_true[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]
        seq_y_mark=self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_true

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_deepar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq
        self.cols=cols
        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        df_raw=pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols=list(df_raw.columns);
            cols.remove(self.target);
            cols.remove('date')
        df_raw=df_raw[['date']+cols+[self.target]]

        num_train=int(len(df_raw)*0.7)  # 70
        num_test=int(len(df_raw)*0.2)  # 20
        num_vali=len(df_raw)-num_train-num_test  # 10
        border1s=[0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]  # [0,65,100-20-5=75]
        border2s=[num_train, num_train+num_vali, len(df_raw)]  # [70,80,100]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=df_raw.columns[1:]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data=df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data=self.scaler.transform(df_data.values)
        else:
            data=df_data.values

        df_stamp=df_raw[['date']][border1:border2]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data[border1:border2]
        if self.inverse:
            self.data_y=df_data.values[border1:border2]
        else:
            self.data_y=data[border1:border2]
        self.data_stamp=data_stamp

    def __getitem__(self, index):
        s_begin=index
        s_end=s_begin+self.seq_len
        r_begin=s_end-self.label_len
        r_end=r_begin+self.label_len+self.pred_len

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]
        seq_y_mark=self.data_stamp[r_begin:r_end]

        return  seq_y, seq_y_mark

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
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
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_ts2vec = sklearn.preprocessing.StandardScaler()

        df_raw=pd.read_csv(os.path.join(self.root_path,  self.data_path),index_col='date', parse_dates=True)
        # dt_embed=_get_time_features(df_raw.index)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[cols+[self.target]]#['date']
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            #cols_data = df_raw.columns[1:]
            df_data = df_raw#[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])
        dt_embed=df_raw.index
        dt_embed=time_features(dt_embed, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        # self.scaler_ts2vec.fit(data_stamp)
        # dt_embed=self.scaler_ts2vec.transform(data_stamp)
        # self.data_ts2vec=np.concatenate([dt_embed[border1:border2] ,self.data_x], axis=-1)
        self.scaler_ts2vec.fit(dt_embed)
        dt_embed=self.scaler_ts2vec.transform(dt_embed)[border1:border2]
        self.data_ts2vec=np.concatenate([dt_embed, self.data_x], axis=-1)
        # data_stamp=data_stamp[border1:border2]
        # self.data_stamp = data_stamp
        self.data_stamp=dt_embed


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,self.data_ts2vec[s_begin:s_end]
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_syn(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq
        self.cols=cols
        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler=StandardScaler()
        # df_raw=pd.read_csv(os.path.join(self.root_path,
        #                                 self.data_path))
        t0=96
        N=7
        # time points
        x=torch.cat(N*[torch.arange(0, t0+24).type(torch.float).unsqueeze(0)])

        # sinuisoidal signal
        A1, A2, A3=60*torch.rand(3, N)
        A4=torch.max(A1, A2)
        fx=torch.cat([A1.unsqueeze(1)*torch.sin(np.pi*x[0, 0:12]/6)+72,
                      A2.unsqueeze(1)*torch.sin(np.pi*x[0, 12:24]/6)+72,
                      A3.unsqueeze(1)*torch.sin(np.pi*x[0, 24:t0]/6)+72,
                      A4.unsqueeze(1)*torch.sin(np.pi*x[0, t0:t0+24]/12)+72], 1)

        # add noise
        fx=fx+torch.randn(fx.shape)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);

        df_raw=df_data=pd.DataFrame(fx.transpose(0,1).numpy())

        num_train=int(len(df_raw)*0.7)  # 70
        num_test=int(len(df_raw)*0.2)  # 20
        num_vali=len(df_raw)-num_train-num_test  # 10
        border1s=[0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]  # [0,65,100-20-5=75]
        border2s=[num_train, num_train+num_vali, len(df_raw)]  # [70,80,100]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]


        if self.scale:
            train_data=df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data=self.scaler.transform(df_data.values)
        else:
            data=df_data.values

        # df_stamp=df_raw[['date']][border1:border2]
        # df_stamp['date']=pd.to_datetime(df_stamp.date)
        # data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data[border1:border2]
        if self.inverse:
            self.data_y=df_data.values[border1:border2]
        else:
            self.data_y=data[border1:border2]
        # self.data_stamp=data_stamp

    def __getitem__(self, index):
        s_begin=index
        s_end=s_begin+self.seq_len
        r_begin=s_end-self.label_len
        r_end=r_begin+self.label_len+self.pred_len

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
        seq_x_mark=None #self.data_stamp[s_begin:s_end]
        seq_y_mark=None #self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_missing_pre_big(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', mr=0.1, cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq
        self.mr=mr
        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __create_missing__(self,df_raw,cols):
        df_missing=df_raw.copy()
        idx=list(range(len(df_missing)))
        for i in range(len(cols)):
            random.seed(i*10)
            s=random.sample(idx, int(len(df_missing)*self.mr))
            s.sort()
            df_missing.loc[s, cols[i]]=np.nan
        for column in list(df_missing.columns[df_missing.isnull().sum()>0]):
            mean_val=df_missing[column].mean()
            df_missing[column].fillna(mean_val, inplace=True)
        return df_missing
    def __read_data__(self):
        self.scaler=StandardScaler()
        self.scaler2=StandardScaler()
        df_raw_1=pd.read_csv('./data/ETT/ETTh1.csv')[0:12*30*24+4*30*24]
        df_raw_2=pd.read_csv('./data/ETT/ETTh2.csv')[0:12*30*24+4*30*24]
        df_raw_3=pd.read_csv('./data/ETT/ETTm1.csv')[0:12*30*24*4+4*30*24*4]
        df_raw_4=pd.read_csv('./data/ETT/ETTm1.csv')[0:12*30*24*4+4*30*24*4]
        df_raw_5=pd.read_csv('./data/exchange_rate/exchange_rate.csv').drop(['4'],axis=1)
        df_raw_5=df_raw_5[0:len(df_raw_5)-int(len(df_raw_5)*0.2)]
        df_raw_6=pd.read_csv('./data/illness/national_illness.csv')
        df_raw_6=df_raw_6[0:len(df_raw_6)-int(len(df_raw_6)*0.2)]
        df_raw_5.columns=df_raw_1.columns
        df_raw_6.columns=df_raw_1.columns
        cols=df_raw_1.columns[1:]
        df_raw_1_missing=self.__create_missing__(df_raw_1,cols)
        df_raw_2_missing=self.__create_missing__(df_raw_2, cols)
        df_raw_3_missing=self.__create_missing__(df_raw_3, cols)
        df_raw_4_missing=self.__create_missing__(df_raw_4, cols)
        df_raw_5_missing=self.__create_missing__(df_raw_5, cols)
        df_raw_6_missing=self.__create_missing__(df_raw_6, cols)
        df_raw_train=pd.concat([df_raw_1[0:int(len(df_raw_1)*0.7)], df_raw_2[0:int(len(df_raw_2)*0.7)], df_raw_3[0:int(len(df_raw_3)*0.7)], df_raw_4[0:int(len(df_raw_4)*0.7)], df_raw_5[0:int(len(df_raw_5)*0.7)], df_raw_6[0:int(len(df_raw_6)*0.7)]])
        df_raw_valid=pd.concat(
            [df_raw_1[int(len(df_raw_1)*0.7):], df_raw_2[int(len(df_raw_2)*0.7):], df_raw_3[int(len(df_raw_3)*0.7):] \
                , df_raw_4[int(len(df_raw_4)*0.7):], df_raw_5[int(len(df_raw_5)*0.7):],
             df_raw_6[int(len(df_raw_6)*0.7):]], axis=0)
        df_raw=pd.concat([df_raw_train,df_raw_valid],axis=0)
        df_raw.index=range(0, len(df_raw))

        df_raw_train_missing=pd.concat([df_raw_1_missing[0:int(len(df_raw_1)*0.7)], df_raw_2_missing[0:int(len(df_raw_2)*0.7)], df_raw_3_missing[0:int(len(df_raw_3)*0.7)], \
                                df_raw_4_missing[0:int(len(df_raw_4)*0.7)], df_raw_5_missing[0:int(len(df_raw_5)*0.7)], df_raw_6_missing[0:int(len(df_raw_6)*0.7)]])
        df_raw_valid_missing=pd.concat(
            [df_raw_1_missing[int(len(df_raw_1_missing)*0.7):], df_raw_2[int(len(df_raw_2_missing)*0.7):], df_raw_3[int(len(df_raw_3_missing)*0.7):] \
                , df_raw_4_missing[int(len(df_raw_4)*0.7):], df_raw_5[int(len(df_raw_5_missing)*0.7):],
             df_raw_6_missing[int(len(df_raw_6)*0.7):]], axis=0)
        df_missing=pd.concat([df_raw_train_missing,df_raw_valid_missing],axis=0)
        df_missing.index=range(0, len(df_raw))


        border1s=[0, int(len(df_raw)*0.7)-self.seq_len]
        border2s=[int(len(df_raw)*0.7), len(df_raw)]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data=cols
            df_data_missing=df_missing[cols_data]
            df_data=df_raw[cols_data]
        elif self.features=='S':
            df_data_missing=df_missing[[self.target]]
            df_data=df_raw[[self.target]]

        if self.scale:
            train_data_missing=df_data_missing[border1s[0]:border2s[0]]
            self.scaler.fit(train_data_missing.values)
            data_missing=self.scaler.transform(df_data_missing.values)
            train_data=df_data[border1s[0]:border2s[0]]
            data=self.scaler.transform(df_data.values)
        else:
            data_missing=df_data_missing.values
            data=df_data.values

        df_stamp=df_raw[['date']]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # dt_embed=df_raw.index
        # dt_embed=time_features(dt_embed, timeenc=self.timeenc, freq=self.freq)

        self.data_x=data_missing[border1:border2]
        self.data_x_true=data[border1:border2]

        if self.inverse:
            self.data_y=df_data_missing.values[border1:border2]
            self.data_y_true=df_data.values[border1:border2]
        else:
            self.data_y=data_missing[border1:border2]
            self.data_y_true=data[border1:border2]
        self.data_stamp=data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin=index  # 0
        s_end=s_begin+self.seq_len  # 96
        r_begin=s_end-self.label_len  # 48
        r_end=r_begin+self.label_len+self.pred_len  # 96+24

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
            seq_y_true=np.concatenate(
                [self.data_x_true[r_begin:r_begin+self.label_len], self.data_y_true[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
            seq_y_true=self.data_y_true[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]  # 0- 96
        seq_y_mark=self.data_stamp[r_begin:r_end]  # 48- 96+24

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_true  # ,self.data_ts2vec[s_begin:s_end]

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_big(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', mr=0.1, cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size==None:
            self.seq_len=24*4*4
            self.label_len=24*4
            self.pred_len=24*4
        else:
            self.seq_len=size[0]
            self.label_len=size[1]
            self.pred_len=size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map={'train': 0, 'val': 1, 'test': 2}
        self.set_type=type_map[flag]

        self.features=features
        self.target=target
        self.scale=scale
        self.inverse=inverse
        self.timeenc=timeenc
        self.freq=freq
        self.mr=mr
        self.root_path=root_path
        self.data_path=data_path
        self.__read_data__()

    def __scale__(self,data,cols):
        data = data[cols]
        self.scaler=StandardScaler()
        self.scaler.fit(data[0:int(len(data)*0.7)].values)
        data=self.scaler.transform(data.values)
        data=pd.DataFrame(data)
        return data

    def __read_data__(self):
        df_raw_1=pd.read_csv('./data/ETT/ETTh1.csv')[0:12*30*24+4*30*24]
        df_raw_2=pd.read_csv('./data/ETT/ETTh2.csv')[0:12*30*24+4*30*24]
        df_raw_3=pd.read_csv('./data/ETT/ETTm1.csv')[0:12*30*24*4+4*30*24*4]
        df_raw_4=pd.read_csv('./data/ETT/ETTm1.csv')[0:12*30*24*4+4*30*24*4]
        df_raw_5=pd.read_csv('./data/exchange_rate/exchange_rate.csv').drop(['4'],axis=1)
        df_raw_5=df_raw_5[0:len(df_raw_5)-int(len(df_raw_5)*0.2)]
        df_raw_6=pd.read_csv('./data/illness/national_illness.csv')
        df_raw_6=df_raw_6[0:len(df_raw_6)-int(len(df_raw_6)*0.2)]
        df_raw_5.columns=df_raw_1.columns
        df_raw_6.columns=df_raw_1.columns
        cols=df_raw_1.columns[1:]
        df_scale_1 = self.__scale__(df_raw_1,cols)
        df_scale_2 = self.__scale__(df_raw_2,cols)
        df_scale_3 = self.__scale__(df_raw_3,cols)
        df_scale_4 = self.__scale__(df_raw_4,cols)
        df_scale_5 = self.__scale__(df_raw_5,cols)
        df_scale_6 = self.__scale__(df_raw_6,cols)

        df_raw_train=pd.concat([df_scale_1[0:int(len(df_scale_1)*0.7)], df_scale_2[0:int(len(df_scale_2)*0.7)], df_scale_3[0:int(len(df_scale_3)*0.7)], df_scale_4[0:int(len(df_scale_4)*0.7)], df_scale_5[0:int(len(df_scale_5)*0.7)], df_scale_6[0:int(len(df_scale_6)*0.7)]])
        df_raw_valid=pd.concat(
            [df_scale_1[int(len(df_scale_1)*0.7):], df_scale_2[int(len(df_scale_2)*0.7):], df_scale_3[int(len(df_scale_3)*0.7):] \
                , df_scale_5[int(len(df_scale_4)*0.7):], df_scale_5[int(len(df_scale_5)*0.7):],
             df_scale_6[int(len(df_scale_6)*0.7):]], axis=0)
        df_raw=pd.concat([df_raw_train,df_raw_valid],axis=0)
        df_raw.index=range(0, len(df_raw))
        data = df_raw.copy()
        data=np.array(data)
        border1s=[0, int(len(df_raw)*0.7)-self.seq_len]
        border2s=[int(len(df_raw)*0.7), len(df_raw)]
        border1=border1s[self.set_type]
        border2=border2s[self.set_type]

        # if self.features=='M' or self.features=='MS':
        #     df_data=df_raw[cols]
        # elif self.features=='S':
        #     df_data=df_raw[[self.target]]

        # if self.scale:
        #     self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
        #     data=self.scaler.transform(df_data.values)
        # else:
        #     data=df_data.values

        df_stamp=pd.concat([df_raw_1,df_raw_2,df_raw_3,df_raw_4,df_raw_5,df_raw_6])[['date']]
        df_stamp['date']=pd.to_datetime(df_stamp.date)
        data_stamp=time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        self.data_x=data[border1:border2]

        # if self.inverse:
        #     self.data_y=df_data.values[border1:border2]
        # else:
        self.data_y=data[border1:border2]
        self.data_stamp=data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin=index  # 0
        s_end=s_begin+self.seq_len  # 96
        r_begin=s_end-self.label_len  # 48
        r_end=r_begin+self.label_len+self.pred_len  # 96+24

        seq_x=self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y=np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y=self.data_y[r_begin:r_end]
        seq_x_mark=self.data_stamp[s_begin:s_end]  # 0- 96
        seq_y_mark=self.data_stamp[r_begin:r_end]  # 48- 96+24

        return seq_x, seq_y, seq_x_mark, seq_y_mark # ,self.data_ts2vec[s_begin:s_end]

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)