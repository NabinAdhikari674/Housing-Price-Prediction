print("Importing Packages and Data ...")
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from lightgbm import LGBMRegressor   # Have to install Package : lightgbm
from catboost import CatBoostRegressor # Have to install Package : catboost
from sklearn.linear_model import Ridge,LinearRegression,Lasso,ElasticNet,RANSACRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from preProcessor import np,Xtrain,Xtest,Ytrain,Ytest,pd
print("Import Done\n")
r_s=21
my_regressors=[
               ElasticNet(),
               CatBoostRegressor(logging_level='Silent'),
               LGBMRegressor(),
               RandomForestRegressor(n_estimators=100),
               AdaBoostRegressor(),
               Ridge(),
               DecisionTreeRegressor(max_depth=3),
               LinearRegression(),
               KNeighborsRegressor(),
               Lasso(),
               KernelRidge(),
               RANSACRegressor(),
              ]

algos = ['Quadratic LR (Degree=2)','ElasticNet','CatBoostRegressor','LGBMRegressor','RandomForestRegressor','AdaBoostRegressor','Rigde Regressor','DecisionTreeRegressor','LinearRegression','KNeighborsRegressor','LASSO','KernelRidge','RANSACRegressor']


trainR2=[]
testR2=[]
trainRMSE=[]
testRMSE=[]

pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
Xquad = quadratic.fit_transform(Xtrain)
pr.fit(Xquad,Ytrain)
predTrain = pr.predict(quadratic.fit_transform(Xtrain))
predTest = pr.predict(quadratic.fit_transform(Xtest))
trainR2.append(r2_score(Ytrain,predTrain))
trainRMSE.append(np.sqrt(mean_squared_error(Ytrain,predTrain)))
testR2.append(r2_score(Ytest,predTest))
testRMSE.append(np.sqrt(mean_squared_error(Ytest,predTest)))

for regressor in my_regressors:
    print("\n\tTraining ",type(regressor).__name__," : ")
    regressor.fit(Xtrain,Ytrain)
    predTrain = regressor.predict(Xtrain)
    predTest = regressor.predict(Xtest)
    trainR2.append(r2_score(Ytrain,predTrain))
    trainRMSE.append(np.sqrt(mean_squared_error(Ytrain,predTrain)))
    testR2.append(r2_score(Ytest,predTest))
    testRMSE.append(np.sqrt(mean_squared_error(Ytest,predTest)))
    print("Training and Testing DONE")


print("\nPreparing Results ...")
df_results=pd.DataFrame({"Algorithms":algos,"Training R2":trainR2,"Training RMSE":trainRMSE,"Testing R2":testR2,"Testing RMSE":testRMSE})

print("\nResults of Different ML Models with different Stats : \n",df_results)
