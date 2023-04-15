import pandas as pd
import xgboost as xgb
from utils.timefeatures import time_features
from utils.tools import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from utils.metrics import metric
import numpy as np
from sklearn.model_selection import GridSearchCV
df=pd.read_csv(r'D:\pycharm\Informer2020-main\data\ETT\ETTh1.csv')
data=df[df.columns[1:]]
train_data=data[0:12*30*24]
scale=StandardScaler()
scale.fit(train_data.values)
data=scale.transform(data.values)

stamp=df[['date']]
stamp['date']=pd.to_datetime(df[['date']].date)
data_stamp=time_features(stamp, timeenc=0, freq='h')

border1s=[0, 12*30*24-96, 12*30*24+4*30*24-96]
border2s=[12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]

x_train=data_stamp[0:12*30*24+4*30*24]
x_test=data_stamp[12*30*24+4*30*24:12*30*24+8*30*24]

y_train=data[0:12*30*24]
# y_train.index=range(len(y_train))
y_test=data[12*30*24+4*30*24:12*30*24+8*30*24]
other_params={'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
              'subsample': 0.8, 'colsample_bytree': 0.3, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
gsc = GridSearchCV(
            estimator=xgb.XGBRegressor(),
            param_grid={"learning_rate": np.linspace(0.01,2,5),
                        'n_estimators': (50,100,200),
                        "max_depth": [ 3, 4, 5, 6, 8],
                        "min_child_weight": [ 1, 3, 5, 7],
                        "gamma":[ 0.0, 0.1, 0.2],
                        'subsample':np.linspace(0.7,0.9,2),
                        "colsample_bytree":np.linspace(0.5,0.98,5),},
            cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
preds=[]
for i in range(len(y_test)):
    print(i)
    multioutputregressor=MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', **other_params)).fit(
        data_stamp[0:12*30*24+4*30*24+i], data[0:12*30*24+4*30*24+i])
    pred=multioutputregressor.predict(x_test[i:i+1])
    preds.append(pred)
preds=np.array(preds)
# multioutputregressor=MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', **other_params)).fit(
#     data_stamp[0:12*30*24+4*30*24], data[0:12*30*24+4*30*24])
# print(multioutputregressor.estimators_[0].best_params_)
# preds=multioutputregressor.predict(x_test)
mae, mse, rmse, mape, mspe = metric(preds, y_test)
print('mae:{}, mse:{},rmse:{},mape:{},mspe:{}'.format(mae, mse, rmse, mape, mspe))
#单步预测mae:0.7502801591120261, mse:1.0514711382042587,rmse:1.0254126672731612,mape:10.419080940804818,mspe:37482.50320660339
#多步预测mae:0.7403640460187894, mse:0.9818527748515067,rmse:0.9908848443949008,mape:14.557290018111358,mspe:66082.18578402884
#{'colsample_bytree': 0.3, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 100}
#mae:0.7733292688666659, mse:0.9853572823177117,rmse:0.9926516419760316,mape:22.541930014645935,mspe:174370.46473709735
# {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
#               'subsample': 0.8, 'colsample_bytree': 0.3, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
#mae:0.7681467069479506, mse:0.974577071759668,rmse:0.9872067016383489,mape:25.372378784577165,mspe:220343.75587318867
# {'colsample_bytree': 0.86, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 8, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.7}
# mae:0.7257083959996443, mse:1.0003983755730024,rmse:1.0001991679525646,mape:6.604968770265778,mspe:17266.83734337709