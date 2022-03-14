import pandas as pd
import matplotlib.pyplot as plt
PATH="/Users/hannaalbright1/Desktop/CSCI 183/"

def readCSV(iFile, cols):
    print("Reading in File: ", iFile)
    df=pd.read_csv(iFile, header=0, delimiter=",", usecols=cols)
    return df
def plotGraphs(df, target): #scatter and line
    for label in df.drop(target, axis=1).columns:
        if df[label].dtype in ['int64', 'float64']:
            df.plot.scatter(y=target, x=label)
            name = PATH+target+label + "graph.png"
            plt.savefig(name)
            plt.close()

sDeaths=readCSV(PATH+"Medical_Examiner-Coroner__Suicide_Deaths_dataset.csv", ["Death Date", "Age", "Race", "Gender"])
sDeaths['Death Date']=pd.to_datetime(sDeaths['Death Date'])
sDeaths.set_index('Death Date', inplace=True)
sDeaths.index=sDeaths.index.normalize()
sDeaths['Race'].replace(['White', 'Hispanic', 'Asian', 'OtherPacificIslander', 'Other','BlackAfricanAmerican', 'Unknown', 'American Indian'], range(0,8), inplace=True)
sDeaths['Gender'].replace(['Male', 'Female', None], [0,1,2], inplace=True)
sDeaths_2020=sDeaths['2020-02':'2021-01-31']
sDeaths_2019=sDeaths['2019-01':'2019-12-31']
sDeaths_2019.to_csv(PATH+"sDeaths2019.csv")
cCases=readCSV(PATH+"COVID-19_case_counts_by_date.csv",["Date", "Total_cases", "New_cases"])
cCases['Date']=pd.to_datetime(cCases['Date'])
cCases.set_index('Date', inplace=True)
c2020=cCases['2020-02':'2021-01-31']#first covid case documented on 2020/1/27 so start at the beginning of the next month
df_2020=sDeaths_2020.join(c2020['Total_cases'], how='left')
df_2020.to_csv(PATH+"sDeaths2020.csv")

plotGraphs(df_2020, "Total_cases")