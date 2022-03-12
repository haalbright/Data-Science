import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
PATH="/Users/hannaalbright1/Desktop/CSCI 183/"


def readCSV(iFile, cols):
    print("Reading in File: ", iFile)
    df=pd.read_csv(iFile, header=0, delimiter=",", usecols=cols)
    return df



def plotGraphs(data, target, corrCol):
    df=data.filter(items=corrCol+[target])
    df2=df.drop(target, axis=1)
    for label in df.drop(target, axis=1).columns:
        df2.drop(label, axis=1, inplace=True)
        for label2 in df2.columns:
            df.plot.scatter(y=label2, x=label, c="target", colormap="viridis")
            name = PATH+target+"/"+label + label2+ "graph.png"
            plt.savefig(name)
            plt.close()
    return df

def logReg(data,target,features):
    x=data.filter(items=features)
    y=data[target].copy()
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25, random_state=123)
    model=LogisticRegression()
    model.fit(xTrain,yTrain)
    yPred = pd.Series(model.predict(xTest))
    yTest = yTest.reset_index(drop=True)
    z = pd.concat([yTest, yPred], axis=1)
    z.columns = ['True', 'Prediction']
    print("Accuracy:", accuracy_score(yTest, yPred))
    print("Precision:", precision_score(yTest, yPred))
    print("Recall:", recall_score(yTest, yPred))
    print("F1:", f1_score(yTest, yPred))
    return z

sDeaths=readCSV(PATH+"Medical_Examiner-Coroner__Suicide_Deaths_dataset.csv", ["Death Date", "Age", "Race", "Gender", "Incident Location", "Incident City", "Cause of Death"])
cCases=readCSV(PATH+"COVID-19_case_counts_by_date.csv",["Date", "Total_cases", "New_cases"])
sDeaths['Death Date']=pd.to_datetime(sDeaths['Death Date'])
cCases['Date']=pd.to_datetime(cCases['Date'])
##first covid case documented on 2020/1/27
## I am saying that covid started on January 1, 2020 and not taking in data from this month as it isn't complete

# heartPlot=plotGraphs(heart, "target", ["age", "trestbps", "chol", "thalach", "oldpeak"])#want only numberical values and target
#
# heartLog=logReg(heart, "target", ["age", "thalach"])
# heartLog=logReg(heart, "target", ["age", "oldpeak"])
# heartLog=logReg(heart, "target", ["thalach", "chol"])
# heartLog=logReg(heart, "target", ["thalach", "trestbps"])
# heartLog=logReg(heart, "target", ["thalach", "oldpeak"])
# heartLog=logReg(heart, "target", ["trestbps", "chol"])
