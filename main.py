import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score as auc
from sklearn import ensemble
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

#### Import Data ###########
df = pd.read_csv('train.csv')
keys  = df.columns

###### Find Variable Significant ########## 
model = ensemble.RandomForestClassifier(n_estimators = 100)
model.fit(df[keys[2:]].values,df[keys[1]].values.ravel())

imprtc = model.feature_importances_
imprtc = pd.DataFrame(imprtc, index=keys[2:], 
                          columns=["Importance"])
imprtc["Std"] = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
x = range(imprtc.shape[0])
y = imprtc.ix[:, 0]
yerr = imprtc.ix[:, 1]
plt.xticks(x, keys[2:],rotation=90)
plt.bar(x, y, yerr=yerr, align="center")
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()

######### Filter out Important Variables ############

model = LassoCV()
rfe1 = RFE(model,25)
rfe2 = rfe1.fit(df[keys[2:]],df[keys[1]]) 
x = rfe2.transform(df[keys[2:]])
new_features = df[keys[2:]].columns[rfe2.get_support()]


######### Create dummy variables ############

dummy = []
for i in df[new_features]:
    ct = df[i].nunique()
    if ct > 2 and ct <=10:
        dummy.append(i)

df_with_dummies = pd.get_dummies( df[keys[2:]], columns = dummy ,drop_first=True )
df_with_dummies = df_with_dummies.drop('target',axis=1)


########## Create Training and Test Sets #################
df_trn_prd,df_test_prd,df_trn_tgt,df_test_tgt = train_test_split(df_with_dummies,df['target'],test_size = 0.2,random_state=0)

########## Standardize the data #########
scaler=StandardScaler().fit(df_trn_prd)
df_trn_prd=scaler.transform(df_trn_prd)
df_test_prd=scaler.transform(df_test_prd)

######### Build an XGB model ################
clf  = xgb.XGBClassifier(n_estimators=120, nthread=-1,missing=-1, max_depth = 3,silent = False,min_child_weight = .5
                         ,learning_rate = .1,subsample=.8,colsample_bytree=0.8,max_delta_step =1,gamma =.2,reg_alpha=0,
                         reg_lambda = .2,base_score=0.3,objective ='binary:logistic',seed=50)
model  = clf.fit(df_trn_prd,df_trn_tgt)


######### Check Accuracy ##################
pred = clf.predict_proba(df_test_prd)[:,1]
result = auc(df_test_tgt, pred,average='macro')
fpr, tpr, thresholds = roc_curve(df_test_tgt, pred)
plt.plot(fpr, tpr, label='xgboost')
plt.title(result,fontsize=20)
fig = plt.gcf()
fig.set_size_inches(16.5, 4.5)
plt.show()



