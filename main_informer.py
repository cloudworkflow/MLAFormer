import argparse
import os
import torch
import time
from exp.exp_informer import Exp_Informer
from exp.exp_missing import Exp_missing
from exp.exp_missing_pre import Exp_missing_pre
from exp.exp_vlength_pre import Exp_vlength_pre
from exp.exp_big_pre import Exp_big_pre
from exp.exp_mae_pre import Exp_mae_pre
import exp
# import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

start_time=time.time()
parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<MLAFormer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD),logsparse,Transformer,Autoformer,Autoformerstack, Auto_local_Informer, ts2vec, ts2vec_former,TFT]')
parser.add_argument('--ts2vec_model', type=str, required=False, default='full',help='model of experiment, options: [informer, informerstack, informerlight(TBD),logsparse,Transformer,Autoformer, Autoformerstack,Auto_local_Informer]')
parser.add_argument('--attn', type=str, default='auto', help='attention used in encoder, options:[prob, full, logsparse,auto,auto_loss,auto_loss_V]')
parser.add_argument('--local_casual', action='store_false',default=True, help='use local attention')
parser.add_argument('--local_casual_cross', action='store_false',default=False, help='cross use local attention')
parser.add_argument('--distil', action='store_false', default=False, help='whether to use distilling in encoder, using this argument means not using distilling')
parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
parser.add_argument('--train_epochs_ts2vec_former', type=int, default=2, help='train epochs in ts2vec_former')
parser.add_argument('--q_len', type=int, default=2, help='attention q_len')
parser.add_argument('--itr', type=int, default=40, help='fexperiments times')
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='data') #custom  syn ETTm1
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file') #exchange_rate national
parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='data file') #national
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=96 , help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--pred_len_full', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--quantiles', type=list, default=[0.1,0.5,0.9], help='quantiles')
parser.add_argument('--quan', action='store_false',default=False, help='use quantiles')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--sub_len', type=int, default=1, help='sub_len of the sparse attention')


parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')#0.05
parser.add_argument('--embed', type=str, default='fixed', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

parser.add_argument('--patience', type=int, default=2, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')#0.0001
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Logsparse Transformer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

parser.add_argument('--n_time_series', type=int, default=7, help='Number of time series present in input')#3
parser.add_argument('--n_head', type=int, default=8, help='Number of heads in the MultiHeadAttention mechanism')
parser.add_argument('--num_layer', type=int, default=2, help='The number of transformer blocks in the model.')
parser.add_argument('--n_embd', type=int, default=512, help='The dimention of Position embedding and time series ID embedding')
parser.add_argument('--forecast_history', type=int, default=96, help='The number of historical steps fed into the time series model')
parser.add_argument('--additional_params', type=dict, default=dict(), help='additional_params')

"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Transformer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
parser.add_argument('--number_time_series', type=int, default=7, help='Number of time series present in input')
parser.add_argument('--seq_length', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--output_seq_len', type=int, default=24, help='The length of your output sequence')
#d_model,n_heads,dropout,
parser.add_argument('--forward_dim', type=int, default=2048, help='Currently not functional')
parser.add_argument('--sigmoid', type=bool, default=False, help='Whether to apply a sigmoid activation to the final')

"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Autoformer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')

"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<pyraformer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
parser.add_argument('-embed_type', type=str, default='DataEmbedding',help='DataEmbedding,CustomEmbedding')
parser.add_argument('-decoder', type=str, default='FC')  # selection: [FC, attention]
parser.add_argument('-seq_num', type=int, default=1)
parser.add_argument('-pretrain', action='store_true', default=False)
parser.add_argument('-hard_sample_mining', action='store_true', default=False)
parser.add_argument('-window_size', type=str, default='[4,4,2]')  # The number of children of a parent node.
parser.add_argument('-inner_size', type=int, default=3)  # The number of ajacent nodes.
# CSCM structure. selection: [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
parser.add_argument('-CSCM', type=str, default='Bottleneck_Construct')
parser.add_argument('-truncate', action='store_true',default=False)  # Whether to remove coarse-scale nodes from the attention structure
parser.add_argument('-use_tvm', action='store_true', default=False)  # Whether to use TVM.
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)
parser.add_argument('-d_inner_hid', type=int, default=512)
parser.add_argument('-d_bottleneck', type=int, default=128)
parser.add_argument('-n_layer', type=int, default=2)
parser.add_argument('-covariate_size', type=int, default=4)

"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<TS2VEC>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
parser.add_argument('--repr_dims', type=int, default=320, help='The representation dimension (defaults to 320)')
parser.add_argument('--max-train-length', type=int, default=3000,help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
parser.add_argument('--save-every', type=int, default=None,help='Save the checkpoint every <save_every> iterations/epochs')
parser.add_argument('--seed', type=int, default=None, help='The random seed')
parser.add_argument('--max-threads', type=int, default=None,help='The maximum allowed number of threads used by this process')
parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
parser.add_argument('--eval', action="store_false", default=True, help='Whether to perform evaluation after training')
parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
parser.add_argument('--temporal_unit', type=int, default=0, help='The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.')
parser.add_argument('--input_dims', type=int, default=14, help='The input dimension. For a univariate time series, this should be set to 1.')
parser.add_argument('--hidden_dims', type=int, default=64, help='The hidden dimension of the encoder.')
parser.add_argument('--depth', type=int, default=10, help='The gpu used for training and inference.')

"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<TFT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
parser.add_argument('--input_obs_loc', type=list, default=[0])
parser.add_argument('--static_input_loc', type=list, default=None)
"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<DeepAR>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
parser.add_argument('--lstm_layers', type=int, default=3, help='lstm_layers')
parser.add_argument('--lstm_hidden_dim', type=int, default=160, help='lstm_hidden_dim')
parser.add_argument('--sampling', action='store_false', help='Whether to sample during evaluation')
parser.add_argument('--sample_times', type=int, default=100, help='sample_times')

"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<missing>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
parser.add_argument('--missing', action='store_false',default=False)
parser.add_argument('--mr', type=float, default=0.3, help='missing ratio')
parser.add_argument('--missing_pre', action='store_false',default=False)
parser.add_argument('--vlength', action='store_false',default=False)
parser.add_argument('--big', action='store_false',default=False)
parser.add_argument('--w', type=float, default=1, help='weight')
parser.add_argument('--mae', action='store_false',default=False)


args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
}
# if args.data in data_parser.keys():
#     data_info = data_parser[args.data]
#     args.data_path = data_info['data']
#     args.target = data_info['T']
#     args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]



def main():
    for ii in range(args.itr):
        print('iter: {}'.format(ii+1))
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_plfull{}_pre{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_localatt{}_{}'.format(args.w,args.model, args.data, args.features,
                    args.seq_len, args.label_len, args.pred_len, args.pred_len_full, args.missing_pre,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
                    args.embed, args.distil, args.mix, args.des,args.local_casual,ii)
        # setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
        # setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
        #             args.seq_len, args.label_len, args.pred_len,
        #             args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.
        #             tor,
        #             args.embed, args.distil, args.mix, args.des, ii)
        if args.model == 'deepar':
            Exp = Exp_DeepAR

        elif args.mae:
            Exp = Exp_mae_pre
        elif args.big:
            Exp = Exp_big_pre
        elif args.vlength:
            Exp = Exp_vlength_pre
        elif args.missing:
            if args.missing_pre:
                Exp = Exp_missing_pre
            else:
                Exp = Exp_missing
        else:
            Exp = Exp_Informer
        exp = Exp(args) # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>'.format(setting))
        if args.model=='xgboost':
            exp.train_xgb(setting)
        else:
            path=os.path.join('./checkpoints_pre/', setting)
            # if (not args.missing_pre) or not os.path.exists(path):
            #     exp.train(setting)
            # exp.train(setting)
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<'.format(setting))
        if args.model=='xgboost':
            exp.test_xgb(setting)
        else:
            exp.train(setting)
            exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        torch.cuda.empty_cache()
    end_time=time.time()
    print("Total time: {}".format(end_time-start_time))
if __name__ == '__main__':
    args.window_size=eval(args.window_size)
    args_data=['ETTh2']#['ETTm2','ETTh2']
    args_data_path=['ETTh2.csv']#['ETTm2.csv','ETTh2.csv']
    args_pred_len=[336]
    args_model=['Autoformer']#,'informer','informer','logsparse']#logsparse第一个参数改
    args_attn=['auto']#,'full','prob','logsparse']
    wei=[1]#[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    for i in range(len(args_data)):
        for k in range(len(args_pred_len)):
            for j in range(len(args_attn)):
                for wi in wei:
                    # if j ==0:
                    #     args.learning_rate=0.0001
                    # else:
                    #     args.learning_rate=0.001
                    args.data=args_data[i]
                    args.data_path=args_data_path[i]
                    args.pred_len=args_pred_len[k]
                    # args.pred_len_full=args_pred_len[k]
                    args.model=args_model[j]
                    args.attn=args_attn[j]
                    args.w=wi
                    print('Args in experiment:')
                    print(args)
                    main()

    # args_root_path=['./data/exchange_rate/']
    # args_data_path=['exchange_rate.csv']
    # args.data='custom'
    # args.enc_in = 8
    # args.dec_in = 8
    # args.c_out = 8
    # for i in range(len(args_root_path)):
    #     for k in range(len(args_pred_len)):
    #         for j in range(len(args_attn)):
    #             args.root_path=args_root_path[i]
    #             args.data_path=args_data_path[i]
    #             args.pred_len=args_pred_len[k]
    #             args.attn=args_attn[j]
    #             print('Args in experiment:')
    #             print(args)
    #             main()
    #
    # args_root_path=['./data/illness/']
    # args_data_path=['national_illness.csv']
    # args.data='custom'
    # args.enc_in = 7
    # args.dec_in = 7
    # args.c_out = 7
    # for i in range(len(args_root_path)):
    #     for k in range(len(args_pred_len)):
    #         for j in range(len(args_attn)):
    #             args.root_path=args_root_path[i]
    #             args.data_path=args_data_path[i]
    #             args.pred_len=args_pred_len[k]
    #             args.attn=args_attn[j]
    #             print('Args in experiment:')
    #             print(args)
    #             main()

