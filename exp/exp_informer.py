from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Custom_bus, Dataset_Pred, Dataset_Custom_syn
from exp.exp_basic import Exp_Basic
#from models.model import  *
from models import Autoformer,Auto_local_Informer,Informer,InformerStack,DecoderTransformer,SimpleTransformer,Autoformerstack
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
from torch.autograd import Variable
import warnings
#from ts2vec_models.datautils  import load_forecast_csv
warnings.filterwarnings('ignore')
from sklearn import metrics


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack': InformerStack,
            'logsparse': DecoderTransformer,
            "Transformer": SimpleTransformer,
            "Autoformer" :Autoformer,
            "Autoformerstack": Autoformerstack,
            "Auto_local_Informer":Auto_local_Informer,
        }

        model=model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _load_forecast_csv(self,name):  # , univar=False):
        data=pd.read_csv(os.path.join(self.args.root_path,self.args.data_path), index_col='date', parse_dates=True)
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
            train_slice=slice(None, int(0.7*len(data)))
            valid_slice=slice(int(0.7*len(data)), int(0.8*len(data)))
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

        # if name in ('ETTh1', 'ETTh2', 'electricity'):
        #     pred_lens = [24, 48, 168, 336, 720]
        # else:
        #     pred_lens = [24, 48, 96, 288, 672]

        return data, train_slice, valid_slice, test_slice, n_covariate_cols



    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'bus':Dataset_Custom_bus,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
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
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def loss_q(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        total_loss=torch.tensor([0.0]).cuda()
        for i, q in enumerate(self.args.quantiles):
            errors=target-y_pred[..., i]
            cur_loss=torch.max((q-1)*errors, q*errors)
            total_loss+=torch.sum(cur_loss)
        return total_loss

    def loss_qi(self, y_pred: torch.Tensor, target: torch.Tensor,i) -> torch.Tensor:
        errors=target-y_pred[..., i]
        q=self.args.quantiles[i]
        loss=torch.max((q-1)*errors, q*errors)
        return torch.sum(loss).item()

    def _select_criterion(self):
        criterion =nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
                if self.args.model=='ts2vec'or self.args.model=='ts2vec_former':
                    loss=self.model.fit(
                        # data_ts2vec
                    )
                else:
                    pred, true = self._process_one_batch(
                        vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,None,'val')
                    if self.args.data=='bus' :
                        true = true[..., -self.args.c_out].unsqueeze(-1)
                        if  not self.args.quan:
                            pred = pred[..., -self.args.c_out].unsqueeze(-1)
                    loss=self.loss_q(pred, true) if self.args.quan else criterion(pred,true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count+=1
                model_optim.zero_grad()
                if self.args.model=='ts2vec' or self.args.model=='ts2vec_former':
                    loss=self.model.fit(
                        # data_ts2vec
                    )
                else:
                    pred, true=self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,epoch,'train')
                    if self.args.data=='bus':
                        true=true[..., -self.args.c_out].unsqueeze(-1)
                        if not self.args.quan:
                            pred=pred[..., -self.args.c_out].unsqueeze(-1)
                    loss=self.loss_q(pred, true) if self.args.quan else criterion(pred, true)
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
            print("Epoch: {} cost time: {}".format(epoch + 1, cost_time))
            # f=open("result_cost_time.txt", 'a')
            # f.write(setting+"  \n")
            # f.write('cost_time:{}'.format(cost_time))
            # f.write('\n')
            # f.write('\n')
            # f.close()
            train_loss = np.average(train_loss)
            # f.write('cost_time:{}'.format(cost_time))
            # f.write('\n')
            # f.write('\n')
            # f.close()
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model


    def test(self, setting):
        self.model.eval()
        if self.args.model=='ts2vec':
            data, train_slice, valid_slice, test_slice, n_covariate_cols=self._load_forecast_csv(self.args.data)
            preds, trues=tasks.eval_forecasting(self.model,data, train_slice, valid_slice, test_slice, n_covariate_cols,self.args.pred_len)

        elif self.args.model=='ts2vec_former':
            data, train_slice, valid_slice, test_slice, n_covariate_cols=self._load_forecast_csv(self.args.data)
            preds, trues=tasks.eval_forecasting_ts2vec_former(self.model, data, self.args)

        else:
            test_data, test_loader=self._get_data(flag='test')
            preds = []
            trues = []
            loss_q = []
            loss_q0=[]
            loss_q1=[]
            loss_q2=[]
            with torch.no_grad():
                for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
                    # batch_x[:,:,-1]+=torch.range(1,batch_x.size(1))/10
                    pred, true = self._process_one_batch(
                        test_data, batch_x, batch_y, batch_x_mark, batch_y_mark,None,'test')
                    pd.DataFrame(true[0, :, 0].cpu().detach().numpy()).to_csv('true.csv')
                    pd.DataFrame(pred[0, :, 0].cpu().detach().numpy()).to_csv('pred.csv')
                    if self.args.data=='bus':
                        true = true[..., -self.args.c_out].unsqueeze(-1)
                        if  not self.args.quan:
                            pred = pred[..., -self.args.c_out].unsqueeze(-1)
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
                    if self.args.quan:
                        loss_q.append(self.loss_q(pred, true).item())
                        loss_q0.append(self.loss_qi(pred, true, 0))
                        loss_q1.append(self.loss_qi(pred, true, 1))
                        loss_q2.append(self.loss_qi(pred, true, 2))
            preds = np.array(preds)  #[32,24,7]的list
            trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        # print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = './results/' + setting +'/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)


        if self.args.quan:
            preds=preds.reshape(-1, preds.shape[-3], preds.shape[-2], preds.shape[-1])  # [2848,24,7]
            trues=trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            loss_q=np.average(loss_q)
            loss_q0=np.average(loss_q0)
            loss_q1=np.average(loss_q1)
            loss_q2=np.average(loss_q2)
            print('loss_q:{}, loss_q_0.1:{}, loss_q_0.5:{}, loss_q_0.9:{}'.format(loss_q.item(), loss_q0.item(), loss_q1.item(), loss_q2.item()))
        else:
            preds=preds.reshape(-1, preds.shape[-2], preds.shape[-1])  # [2848,24,7]
            trues=trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe = metric(preds[..., 1] if self.args.quan else preds, trues)
        print('mae:{}, mse:{},rmse:{},mape:{},mspe:{}'.format(mae, mse, rmse, mape, mspe))
        return


    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,epoch,flag):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.model=='logsparse' :
            outputs=self.model(batch_x)
        elif self.args.model=='Transformer':
            outputs=greedy_decode(self.model, batch_x, 24, batch_y[:,-self.args.pred_len:,:])
        elif self.args.model=='pyraformer':
            if flag!='train':
                if self.args.decoder=='FC':
                    # Add a predict token into the history sequence
                    predict_token=torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=batch_x.device)
                    batch_x=torch.cat([batch_x, predict_token], dim=1)
                    batch_x_mark=torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
                outputs=self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
            else:
                if self.args.decoder=='attention':
                    if self.args.pretrain and epoch<1:
                        outputs=self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, True)
                        batch_y=torch.cat([batch_x, batch_y], dim=1)
                    else:
                        outputs=self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
                elif self.args.decoder=='FC':
                    # Add a predict token into the history sequence
                    predict_token=torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=batch_x.device)
                    batch_x=torch.cat([batch_x, predict_token], dim=1)
                    batch_x_mark=torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
                    outputs=self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
        else:#Informer,Autoformer
            outputs = self.model(batch_x,batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)  #[32,24,7]
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)      #[32,24,7]

        return outputs[:, -self.args.pred_len:, :], batch_y

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """ Generates a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode(
        model,
        src: torch.Tensor,
        max_len: int,
        real_target: torch.Tensor,
        unsqueeze_dim=1,
        output_len=1,
        device='cpu',
        multi_targets=1,
        probabilistic=False,
        scaler=None):
    """
    Mechanism to sequentially decode the model
    :src The Historical time series values
    :real_target The real values (they should be masked), however if you want can include known real values.
    :returns torch.Tensor
    """
    src = src.float()
    real_target = real_target.float()
    if hasattr(model, "mask"):
        src_mask = model.mask #[1,30,3]
    memory = model.encode_sequence(src, src_mask)  #[30,1,128]
    # Get last element of src array to forecast from
    ys = src[:, -1, :].unsqueeze(unsqueeze_dim)  #[1,1,3]
    for i in range(max_len):
        mask = generate_square_subsequent_mask(i + 1).to(device) #i=2 [3,3]
        with torch.no_grad():
            out = model.decode_seq(memory,
                                   Variable(ys,requires_grad=True),
                                   Variable(mask,requires_grad=True), i + 1)
            real_target[:, i, 0] = out[:, i]
            src = torch.cat((src, real_target[:, i, :].unsqueeze(1).cuda()), 1)  #[1,31,3]
            ys = torch.cat((ys, real_target[:, i, :].unsqueeze(1).cuda()), 1) #[1,2,3]
        memory = model.encode_sequence(src[:, i + 1:, :], src_mask)
    return ys[:, 1:, :]