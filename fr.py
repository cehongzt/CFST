import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
train= pd.read_csv('F:\\database\\cfst\\lincacfst.csv')
dem=train[["D (mm)","Le (mm)","fy (MPa)","fc (MPa)"]].values
object=train[["N Test (kN)"]].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_trian,y_test=train_test_split(dem,object,test_size=0.3,shuffle=True)
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
kf=KFold(n_splits=10)
rmse=[]
for train_indices,test_indices in kf.split(dem):
    X_train,X_test=dem[train_indices],dem[test_indices]
    Y_train, Y_test = object[train_indices], object[test_indices]
    RLR=RandomForestRegressor()#引入模型
    RLR.fit(X_train,Y_train)#模型训练
    y_pred = RLR.predict(X_test)
    RSM=np.sqrt(mean_squared_error(Y_test,((y_pred))))
    rmse.append(RSM)
print(rmse)
print(f'average rmse:{np.mean(rmse)}')
