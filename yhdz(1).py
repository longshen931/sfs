# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 22:51:13 2021

@author: liujinli
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from tqdm import tqdm
import os
import random
import warnings
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")
def seed_everything(seed=555):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
seed_everything()

df = pd.read_csv('清洗_2018Spring.csv')
df = shuffle(df)

train_df = df[:-14]
valid_df = df[-14:]
# print(train_df)
# print(valid_df)


train_y = train_df.pop('TN') 
train_x = train_df.values
valid_y = valid_df.pop('TN')
valid_x = valid_df.values
 
lgb = LGBMRegressor()
lgb.fit(train_x,train_y)
pred = lgb.predict(valid_x)
# print('score:', mean_squared_error(valid_y,pred))

# xgb = XGBRegressor()
# xgb.fit(train_x,train_y)
# pred = xgb.predict(valid_x)
# print('score:', mean_squared_error(valid_y,pred))

rf = RandomForestRegressor()
rf.fit(train_x,train_y)
pred = rf.predict(valid_x)
# print('score:', mean_squared_error(valid_y,pred))

f, ax = plt.subplots(figsize=(7, 5))
ax.bar(range(len(rf.feature_importances_)),rf.feature_importances_)
ax.set_title("Feature Importances")
f.show()
# print(len(train_df.columns))
# print(len(rf.feature_importances_))
df_show = pd.DataFrame({'f_name':train_df.columns,'importance':rf.feature_importances_})
# print(df_show.sort_values('importance',ascending=False))
df_show = df_show.sort_values('importance',ascending=False)['f_name'].values
best_mse = 100
best_fnum = 4
plt.show()
plt.close()

df_show = pd.DataFrame({'f_name':train_df.columns,'importance':rf.feature_importances_})
# print(df_show.sort_values('importance',ascending=False))
df_show = df_show.sort_values('importance',ascending=True)
plt.show()

f, ax = plt.subplots(figsize=(15, 20))
print(df_show['importance'].values)
ax.barh(df_show['f_name'],df_show['importance'].values)
ax.set_title("Feature Importances")
f.show()
plt.show()
df_show = df_show.sort_values('importance',ascending=False)['f_name'].values

mse=[];r2=[]
for i in range(4,60):
    choose_feature = df_show[:i]
    train_x = train_df[choose_feature].values
    valid_x = valid_df[choose_feature].values
    lgb = LGBMRegressor()
    lgb.fit(train_x,train_y)
    lgb_pred = lgb.predict(valid_x)
   
    # rf = RandomForestRegressor()
    # rf = ElasticNet()
    # rf.fit(train_x,train_y)
    # rf_pred = rf.predict(valid_x)
    pred = lgb_pred
    mse.append( mean_squared_error(valid_y,pred))
    r2.append(r2_score(valid_y,pred))
    
    
    # print(f'n_num:{i},score:{mse}')
    if(best_mse > mean_squared_error(valid_y,pred)):
       best_mse = mean_squared_error(valid_y,pred)
       best_fnum = i
       
print(f'best f_num:{best_fnum}, best mse:{best_mse}')

plt.plot(range(4,60), mse)
plt.title('feature performance')
plt.xlabel('feature number')
plt.ylabel('mse')
plt.show()
plt.close()

plt.plot(range(4,60), r2)
plt.title('feature performance')
plt.xlabel('feature number')
plt.ylabel('r2')
plt.show()
plt.close()

choose_feature = df_show[:best_fnum]
train_x = train_df[choose_feature].values
valid_x = valid_df[choose_feature].values
#min_child_samples=10,reg_alpha=0.03,reg_lambda=0
alpha=[];lamda=[];mse_loss=[];r2_loss=[]
for i in [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]:
    for j in [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]:
        lgb = LGBMRegressor(min_child_samples=10,reg_alpha=i,reg_lambda=j)
        lgb.fit(train_x,train_y)
        alpha.append(i)
        lamda.append(j)
        pred = lgb.predict(valid_x)
        # model = AdaBoostRegressor(lgb,n_estimators=i)
        # model.fit(train_x,train_y)
        
        # pred = model.predict(valid_x)
        mse = mean_squared_error(valid_y,pred)
        mse_loss.append(mse)
        r2 = r2_score(valid_y,pred)
        r2_loss.append(r2)
        #print(f'min_child_samples:{i},min_child_weights:{j},mse_score:{mse},r2_score:{r2}')
# print(df_show)
param_grid =[
    {'max_depth':range(3,12),
     'min_child_weight':range(4,32,4),
    'reg_alpha':[x/100 for x in range(1,51,2)],
    'reg_lambda':[x/100 for x in range(1,51,2)],
    }
]
model = LGBMRegressor()

from sklearn.model_selection import GridSearchCV
print('grid search begin')
grid_search = GridSearchCV(model,param_grid,scoring='neg_mean_squared_error')

grid_search.fit(train_x,train_y)

print(f'best score:{grid_search.best_score_},best param:{grid_search.best_params_}')



def get_pic(model_name,show_name):
    print(f'---------------{model_name} best params is searching-------------')
    if(model_name=='lgb'):  
        u = [x/100 for x in range(1,51)]
        v =  [x/100 for x in range(1,51)]
    elif(model_name == 'lasso'):
        u = [x/100 for x in range(1,51)]
        v =  [x/1000000 for x in range(1,51)]
    elif(model_name=='svr'):
        u = [x for x in range(1,51)]
        v =  [x/100000 for x in range(1,51)]
    elif(model_name=='xgboost'):
       u = [x/100 for x in range(1,51)]
       v =  [x/100 for x in range(1,51)]
        
    u, v = np.meshgrid(u, v)
    print(u.shape,v.shape)
    
    best_mse_i, best_mse_j, best_mse, best_r2 = 0, 0, 1000, 0
    z = np.zeros_like(u)
    z2=np.zeros_like(u)
    print(z.shape)
    for i in tqdm(range(len(u))):
        for j in range(len(u[i])):
            if(model_name=='lgb'):
                model = LGBMRegressor(min_child_samples=10,reg_alpha=u[i][j],reg_lambda=v[i][j])
            elif(model_name=='lasso'):
                model = Lasso(alpha=u[i][j],tol=v[i][j])
            elif(model_name =='svr'):
                model = SVR(C=u[i][j],tol=v[i][j])
            elif(model_name=='xgboost'):
                model=XGBRegressor(max_depth=2,min_child_weight=28,reg_alpha=u[i][j],reg_lambda=v[i][j])
                
            model.fit(train_x,train_y)
            pred = model.predict(valid_x)
            # model = AdaBoostRegressor(lgb,n_estimators=i)
            # model.fit(train_x,train_y)
            
            # pred = model.predict(valid_x)
            mse = mean_squared_error(valid_y,pred)
            r2=r2_score(valid_y,pred)
            z[i][j] = mse
            z2[i][j]=r2
            
            if(best_mse > mse):
                best_mse = mse
                best_mse_i = i
                best_mse_j = j
                best_r2 = r2
    print('---------------------------------------')
    # plt.figure()
    
    # ax = Axes3D(fig)
    plt.ion()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(model_name)
    if(model_name=='lgb'):
        ax.set_xlabel('alpha')
        ax.set_ylabel('lambda')
        print(f'reg_alpha={u[best_mse_i][best_mse_j]},reg_lambda={v[best_mse_i][best_mse_j]},best mse:{best_mse},best r2:{best_r2}')
    elif(model_name=='lasso'):
        ax.set_xlabel('alpha')
        ax.set_ylabel('tol')
        print(f'alpha={u[best_mse_i][best_mse_j]},tol={v[best_mse_i][best_mse_j]},best mse:{best_mse},best r2:{best_r2}')
    elif(model_name =='svr'):
        ax.set_xlabel('C')
        ax.set_ylabel('tol')
        print(f'C={u[best_mse_i][best_mse_j]},tol={v[best_mse_i][best_mse_j]},best mse:{best_mse},best r2:{best_r2}')
    elif(model_name =='xgboost'):
        ax.set_xlabel('reg_alpha')
        ax.set_ylabel('reg_lambda')
        print(f'reg_alpha={u[best_mse_i][best_mse_j]},reg_lambda={v[best_mse_i][best_mse_j]},best mse:{best_mse},best r2:{best_r2}')
    
    if(show_name == 'mse'):
        ax.set_zlabel('mse') 
        surf=ax.plot_surface(u, v, z, cmap='jet')
        fig.colorbar(surf, shrink=0.4, aspect=6)
        plt.show()
    else:
        ax.set_zlabel('r2')
      
        surf=ax.plot_surface(u, v, z2, cmap='jet')
        fig.colorbar(surf, shrink=0.4, aspect=6)
        plt.show()
    # ax.close()
    ax.cla()
    plt.cla()
    plt.close('all')

      
get_pic('lgb','mse')
get_pic('lasso','mse')
get_pic('xgboost','mse')
get_pic('svr','mse')

get_pic('lgb','r2')
get_pic('lasso','r2')
get_pic('xgboost','r2')
get_pic('svr','r2')

z=[];z2=[]
def get_2dpic(model_name,show_name):
    plt.title(model_name)
    z=[];z2=[]
    if(model_name=='lgb'):  
        u = [x/100 for x in range(1,51)]
        v =  [x/100 for x in range(1,51)]
    elif(model_name == 'lasso'):
        u = [x/100 for x in range(1,51)]
        v =  [x/1000000 for x in range(1,51)]
    elif(model_name=='svr'):
        u = [x for x in range(1,51)]
        v =  [x/100000 for x in range(1,51)]
    elif(model_name=='xgboost'):
       u = [x/100 for x in range(1,51)]
       v =  [x/100 for x in range(1,51)]
      
    best_mse_i, best_mse_j, best_mse, best_r2 = 0, 0, 1000, 0
    if show_name=='mse':
        plt.ylabel('mse')
        for i in u:
            if(model_name=='lgb'):
                model = LGBMRegressor(min_child_samples=10,reg_alpha=i)
                plt.xlabel('reg_alpha')
            elif(model_name=='lasso'):
                model = Lasso(alpha=i)
                plt.xlabel('alpha')
            elif(model_name =='svr'):
                model = SVR(C=i)
                plt.xlabel('c')
            elif(model_name=='xgboost'):
                plt.xlabel('reg_alpha')
                model=XGBRegressor(max_depth=2,min_child_weight=28,reg_alpha=i)
            
            model.fit(train_x,train_y)
            pred = model.predict(valid_x)
            mse = mean_squared_error(valid_y,pred)
            r2=r2_score(valid_y,pred)
            z.append(mse)
            z2.append(r2)
            
        plt.plot(u,z)
        min_indx=np.argmin(z)    
        plt.plot(u[min_indx],z[min_indx],'ks')
        show_max='['+str(np.round((u[min_indx]),2))+' '+str(np.round((z[min_indx]),3))+']'
        plt.annotate(show_max,xytext=(u[min_indx],z[min_indx]),xy=(u[min_indx],z[min_indx]))
        plt.show()
        plt.close()
    elif show_name=='r2':
        plt.ylabel('r2')
        for j in v:
           if(model_name=='lgb'):
               model = LGBMRegressor(min_child_samples=10,reg_lambda=j)
               plt.xlabel('reg_lambda')
           elif(model_name=='lasso'):
               model = Lasso(tol=j)
               plt.xlabel('tol')
           elif(model_name =='svr'):
               model = SVR(tol=j)
               plt.xlabel('tol')
           elif(model_name=='xgboost'):
               model=XGBRegressor(max_depth=2,min_child_weight=28,reg_lambda=j)
               plt.xlabel('reg_lambda')
           
           model.fit(train_x,train_y)
           pred = model.predict(valid_x)
           mse = mean_squared_error(valid_y,pred)
           r2=r2_score(valid_y,pred)
           z.append( mse)
           z2.append(r2)
        plt.plot(v,z2)
        max_indx=np.argmax(z2)    
        plt.plot(v[max_indx],z2[max_indx],'ks')
        show_max='['+str(np.round(v[max_indx],2))+' '+str(np.round(z2[max_indx],3))+']'
        plt.annotate(show_max,xytext=(v[max_indx],z2[max_indx]),xy=(v[max_indx],z2[max_indx]))
        plt.show()
        plt.close()
        
get_2dpic('lgb','mse')
get_2dpic('lasso','mse')
get_2dpic('xgboost','mse')
get_2dpic('svr','mse')

get_2dpic('lgb','r2')
get_2dpic('lasso','r2')
get_2dpic('xgboost','r2')
get_2dpic('svr','r2')

        



# plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(u,v,z2,cmap='jet')
# plt.show()
model = LGBMRegressor(min_child_samples=10)
model.fit(train_x,train_y)
def get_pred(model,test_df):
    test_x = test_df[choose_feature].values
    test_pred = model.predict(test_x)
    return test_pred
test_df = pd.read_csv('201809.csv')
get_pred(model,test_df)