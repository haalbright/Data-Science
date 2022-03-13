import pandas as pd
PATH="/Users/hannaalbright1/Desktop/CSCI 183/"

def readCSV(iFile, cols):
    print("Reading in File: ", iFile)
    df=pd.read_csv(iFile, header=0, delimiter=",", usecols=cols)
    return df

sDeaths=readCSV(PATH+"Medical_Examiner-Coroner__Suicide_Deaths_dataset.csv", ["Death Date", "Age", "Race", "Gender"])
sDeaths['Death Date']=pd.to_datetime(sDeaths['Death Date'])
sDeaths.set_index('Death Date', inplace=True)
sDeaths_2020=sDeaths['2020-02':'2021-01-31']
sDeaths_2019=sDeaths['2019-01':'2019-12-31']
s2020=pd.get_dummies(sDeaths_2020)
sresample=s2020.drop("Age", axis=1).resample("M").sum()
sresample['Avg_Age']=s2020['Age'].resample("M").mean()

cCases=readCSV(PATH+"COVID-19_case_counts_by_date.csv",["Date", "Total_cases", "New_cases"])
cCases['Date']=pd.to_datetime(cCases['Date'])
cCases.set_index('Date', inplace=True)
c2020=cCases['2020-02':'2021-01-31']#first covid case documented on 2020/1/27 so start at the beginning of the next month
cresample=c2020.drop("Total_cases", axis=1).resample("M").sum()
df_2020=cresample.join(sresample)
df_2020.to_csv(PATH+"s2020.csv")

s2019=pd.get_dummies(sDeaths_2019)
sresample2=s2019.drop("Age", axis=1).resample("M").sum()
sresample2['Avg_Age']=s2019['Age'].resample("M").mean()
sresample2.to_csv(PATH+"s2019.csv")
