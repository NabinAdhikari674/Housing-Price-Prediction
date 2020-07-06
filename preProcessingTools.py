#Custom Pre Processing Tools Built by NabinAdhikeri674

def removeHighNullColumns(data):
    nullPercentage = data.isnull().sum() * 100 / len(data)
    print("## Input Data Shape : ",data.shape)
    print("Removing Following Columns Due To High Null Percentage (higher than 40%) :")
    for index,null in zip(nullPercentage.index,nullPercentage):
        if null>40:
            data = data.drop([index],axis=1)
            print(index," : ",null)
    print("## Output Data Shape : ",data.shape)
    return data

def removeTargetNulls(data,target=""):
    print("## Input Data Shape : ",data.shape)
    import pandas as pd
    print("Removing Rows with Null or Empty Target values : ")
    if target == "":
        dataX = data.iloc[:,:-1]
        dataY = data.iloc[:,-1]
    else:
        dataX = data.drop([target],axis=1)
        dataY = data[target]
    dataX = pd.DataFrame(dataX)
    dataY = pd.DataFrame(dataY)
    nulls = dataY.loc[pd.isna(dataY.iloc[:,0]), :].index
    count = 0
    for i in nulls:
        data = data.drop([i],axis=0)
        count += 1
    print(count," Rows Deleted from Input Data")
    print("## Output Data Shape : ",data.shape)
    return data

def fillNullValues(data):
    print("Filling Null Values for Input Data")
    dataCat = data.select_dtypes("object")
    for column in dataCat.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    dataNum = data.select_dtypes(["float64","int64"])
    for column in dataNum.columns:
        data[column].fillna(data[column].median(), inplace=True)
    data = dataNum.merge(dataCat,left_index=True,right_index=True).reset_index(drop=True)
    return data

def removeLowVarianceData(data,target):
    print("## Input Data Shape : ",data.shape)
    print("(Inverse Variance i.e. Higher is Lower Variance)")
    rows,__ = data.shape
    dataCat = data.select_dtypes("object")
    dataNum = data.select_dtypes(["float64","int64"]).drop(target,axis=1)
    returnData = data.loc[:,:]
    def columnRemover(returnData,data,type="categorical"):
        if type == "categorical":
            limit = 95
            print("Processing Categorical Data :: Data Shape Now : ",returnData.shape)
        elif type == "numerical" :
            limit = 98
            print("Processing Numerical Data :: Data Shape Now : ",returnData.shape)
        else:
            limit = 90
        for column in data.columns:
            df = data[column].value_counts()
            if (any( limit <= val*100/rows for val in df.values)) :
                print(column," : ",end="")
                returnData = returnData.drop([column],axis=1)
                #print(df.index)
                #print(df.values)
                print("  Inverse Variance : ",(df.values *100/ rows).max(),"%")
        return returnData
    returnData = columnRemover(returnData,dataCat,"categorical")
    returnData = columnRemover(returnData,dataNum,"numerical")
    print("## Output Data Shape : ",returnData.shape)
    return returnData

def scatterPlotData(data,target):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cols=['LSTAT','INDUS','NOX','RM','MEDV']
    sns.pairplot(data[cols],height=2)
    plt.tight_layout()
    plt.show()

def correlationHeatmap(data):
    import matplotlib.pyplot as plt
    import seaborn as sns

    correlations = data.corr()
    #correlations = correlations.nlargest(15,'SalePrice')['SalePrice'].index
    #correlations = data[correlations].corr()
    fig, ax = plt.subplots(figsize=(30,30))
    heatmap = sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=1, annot=True, cbar_kws={"shrink": 0.70})
    heatmap = heatmap.get_figure()
    heatmap.savefig('heatmap.png', dpi=400)
    plt.show();
