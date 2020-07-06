print("\n\t\t\t\t###### Running preProcessor.py ######")
print("Importing Packages...##",end=" ")
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import preProcessingTools as ppt
print(" IMPORT DONE.\n")

data=pd.read_excel('houseData.xls',header=None)
data.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data=shuffle(data)
data = ppt.removeHighNullColumns(data)
data = ppt.removeTargetNulls(data,"MEDV")
data = ppt.fillNullValues(data)
data = ppt.removeLowVarianceData(data,"MEDV")
print("Setting Inputs and Targets ... ##",end='')
inputs=data.iloc[:,:-1]
targets=data['MEDV']
print(" ## DONE .\n")

chs = input ("\t Display Plots of Different Columns of the data ? [ y / n ] (Input \"y\" for Yes AND \"n\" for No)")
if chs == "y" :
    ppt.scatterPlotData(data,"MEDV")
    chs =""
chs = input ("\t Display Correlation HeatMap of the data ? [ y / n ] (Input \"y\" for Yes AND \"n\" for No)")
if chs == "y":
    ppt.correlationHeatmap(data)

def choser(inputs,targets):
    percent=int(input("\n\t\tEnter the TEST size(IN PERCENTAGE) : "))
    print("\nRandomly Choosing Training and Test Data...##",end=' ')
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(inputs,targets,test_size=percent,random_state=21)
    print(" The Training & Test Data ARE Split")
    return x_train,x_test,y_train,y_test

Xtrain,Xtest,Ytrain,Ytest = choser(inputs,targets)

print("\n\t##Data PreProcessing Done.")
print("\t\t !! Exiting preProcessor.py !!\n")
