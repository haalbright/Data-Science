import pandas as pd
import matplotlib.pyplot as plt
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
# def perMonth(df1, df2, feature):
#     for rng in pd.date_range('2020-02','2021-01-31', )

sDeaths=readCSV(PATH+"Medical_Examiner-Coroner__Suicide_Deaths_dataset.csv", ["Death Date", "Age", "Race", "Gender"])
sDeaths['Death Date']=pd.to_datetime(sDeaths['Death Date'])
sDeaths.set_index('Death Date', inplace=True)
sDeaths_2020=sDeaths['2020-02':'2021-01-31']
# sDeaths_2020['Race'].replace(['White', 'Hispanic', 'Asian', 'OtherPacificIslander', 'Other','BlackAfricanAmerican', 'Unknown', 'American Indian'], range(0,8), inplace=True)
# sDeaths_2020['Gender'].replace(['Male', 'Female', None], [0,1,2], inplace=True)
s2020=pd.get_dummies(sDeaths_2020, dummy_na=True)

cCases=readCSV(PATH+"COVID-19_case_counts_by_date.csv",["Date", "Total_cases", "New_cases"])
cCases['Date']=pd.to_datetime(cCases['Date'])
cCases.set_index('Death Date', inplace=True)
cCases_2020=cCases['2020-02':'2021-01-31']#first covid case documented on 2020/1/27 so start at the beginning of the next month
