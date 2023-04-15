from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models import  Informer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import sklearn.preprocessing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import warnings

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict={
            'informer': Informer,

        }
        model=model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model=nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _load_forecast_csv(self, name):  # , univar=False):
        data=pd.read_csv(os.path.join(self.args.root_path, self.args.data_path), index_col='date', parse_dates=True)
        dt_embed=np.stack([
            data.index.minute.to_numpy(),
            data.index.hour.to_numpy(),
            data.index.dayofweek.to_numpy(),
            data.index.day.to_numpy(),
            data.index.dayofyear.to_numpy(),
            data.index.month.to_numpy(),
            data.index.weekofyear.to_numpy(),
        ], axis=1).astype(np.float)
        n_covariate_cols=dt_embed.shape[-1]

        # if univar:
        #     if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
        #         data = data[['OT']]
        #     elif name == 'electricity':
        #         data = data[['MT_001']]
        #     else:
        #         data = data.iloc[:, -1:]

        data=data.to_numpy()
        if name=='ETTh1' or name=='ETTh2':
            train_slice=slice(None, 12*30*24)
            valid_slice=slice(12*30*24, 16*30*24)
            test_slice=slice(16*30*24, 20*30*24)
        elif name=='ETTm1' or name=='ETTm2':
            train_slice=slice(None, 12*30*24*4)
            valid_slice=slice(12*30*24*4, 16*30*24*4)
            test_slice=slice(16*30*24*4, 20*30*24*4)
        else:
            train_slice=slice(None, int(0.6*len(data)))
            valid_slice=slice(int(0.6*len(data)), int(0.8*len(data)))
            test_slice=slice(int(0.8*len(data)), None)

        scaler=sklearn.preprocessing.StandardScaler().fit(data[train_slice])
        data=scaler.transform(data)
        if name in ('electricity'):
            data=np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
        else:
            data=np.expand_dims(data, 0)

        if n_covariate_cols>0:
            dt_scaler=sklearn.preprocessing.StandardScaler().fit(dt_embed[train_slice])
            dt_embed=np.expand_dims(dt_scaler.transform(dt_embed), 0)
            data=np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data],
                                axis=-1)  # [1,17420,7]  [1,17420,14]
        return data, train_slice, valid_slice, test_slice, n_covariate_cols

    def _get_data(self, flag):
        args=self.args

        data_dict={
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data=data_dict[self.args.data]
        timeenc=0 if args.embed!='timeF' else 1

        if flag=='test':
            shuffle_flag=False;
            drop_last=True;
            batch_size=args.batch_size;
            freq=args.freq
        elif flag=='pred':
            shuffle_flag=False;
            drop_last=False;
            batch_size=1;
            freq=args.detail_freq
            Data=Dataset_Pred
        else:
            shuffle_flag=True;
            drop_last=True;
            batch_size=args.batch_size;
            freq=args.freq
        data_set=Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader=DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim=optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion=nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss=[]
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, data_ts2vec) in enumerate(vali_loader):
                pred, true=self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss=criterion(pred, true)
                total_loss.append(loss.item())

        total_loss=np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader=self._get_data(flag='train')
        vali_data, vali_loader=self._get_data(flag='val')
        test_data, test_loader=self._get_data(flag='test')

        path=os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now=time.time()

        train_steps=len(train_loader)
        early_stopping=EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim=self._select_optimizer()
        criterion=self._select_criterion()

        if self.args.use_amp:
            scaler=torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count=0
            train_loss=[]

            self.model.train()
            epoch_time=time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, data_ts2vec) in enumerate(train_loader):
                iter_count+=1
                model_optim.zero_grad()
                pred, true=self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss=criterion(pred, true)
                train_loss.append(loss.item())

                if (i+1)%100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i+1, epoch+1, loss.item()))
                    speed=(time.time()-time_now)/iter_count
                    left_time=speed*((self.args.train_epochs-epoch)*train_steps-i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count=0
                    time_now=time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.requires_grad_(True)
                    loss.backward()
                    model_optim.step()

            cost_time=time.time()-epoch_time
            print("Epoch: {} cost time: {}".format(epoch+1, cost_time))
            # f=open("result_cost_time.txt", 'a')
            # f.write(setting+"  \n")
            # f.write('cost_time:{}'.format(cost_time))
            # f.write('\n')
            # f.write('\n')
            # f.close()
            train_loss=np.average(train_loss)

            vali_loss=self.vali(vali_data, vali_loader, criterion)
            test_loss=self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch+1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path=path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):

        self.model.eval()
        test_data, test_loader=self._get_data(flag='test')
        preds=[]
        trues=[]
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, data_ts2vec) in enumerate(test_loader):
                pred, true=self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        preds=np.array(preds)
        trues=np.array(trues)

        preds=preds.reshape(-1, preds.shape[-2], preds.shape[-1])  # [2848,24,7]
        trues=trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = './results/' + setting +'/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe=metric(preds, trues)
        print('mae:{}, mse:{},rmse:{},mape:{},mspe:{}'.format(mae, mse, rmse, mape, mspe))
        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mae:{}, mse:{},rmse:{},mape:{},maspe:{}'.format(mae, mse, rmse, mape, mspe))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        # np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path+'pred.npy', preds)
        # np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader=self._get_data(flag='pred')

        if load:
            path=os.path.join(self.args.checkpoints, setting)
            best_model_path=path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds=[]
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                pred, true=self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                # [1,24,7]
                preds.append(pred.detach().cpu().numpy())

        preds=np.array(preds)
        preds=preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        # folder_path = './results/' + setting +'/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #
        # np.save(folder_path+'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x=batch_x.float().to(self.device)
        batch_y=batch_y.float()

        batch_x_mark=batch_x_mark.float().to(self.device)
        batch_y_mark=batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp=torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp=torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp=torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs=self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs=self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs=self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs=self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs=dataset_object.inverse_transform(outputs)  # [32,24,7]
        f_dim=-1 if self.args.features=='MS' else 0
        batch_y=batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # [32,24,7]

        return outputs, batch_y

