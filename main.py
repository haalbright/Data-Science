import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, \
    precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def readCSV(iFile, cols):
    print("Reading in File: ", iFile)
    df = pd.read_csv(iFile, header=0, delimiter=",", usecols=cols)
    return df


sDeaths = readCSV("Medical_Examiner-Coroner__Suicide_Deaths_dataset.csv",
                  ["Death Date", "Age", "Race", "Gender"])
sDeaths['Death Date'] = pd.to_datetime(sDeaths['Death Date'])
sDeaths.set_index('Death Date', inplace=True)
sDeaths.index = sDeaths.index.normalize()
# total num of deaths
# sDeaths['Death Date'] = pd.to_datetime(sDeaths['Death Date'])
sDeaths['Race'].replace(
    ['White', 'Hispanic', 'Asian', 'OtherPacificIslander', 'Other',
     'BlackAfricanAmerican', 'Unknown', 'American Indian'], range(0, 8),
    inplace=True)
sDeaths['Gender'].replace(['Male', 'Female', None], [0, 1, 2], inplace=True)
sDeaths_2020 = sDeaths['2020-02':'2021-01-31']
sDeaths_2019 = sDeaths['2019-01':'2019-12-31']
sDeaths_2019.to_csv("sDeaths2019.csv")

cCases = readCSV("COVID-19_case_counts_by_date.csv",
                 ["Date", "Total_cases", "New_cases"])
cCases['Date'] = pd.to_datetime(cCases['Date'])
cCases.set_index('Date', inplace=True)
c2020 = cCases[
        '2020-02':'2021-01-31']  # first covid case documented on 2020/1/27 so start at the beginning of the next month
df_2020 = sDeaths_2020.join(c2020['Total_cases'], how='left')
df_2020.to_csv("sDeaths2020.csv")


def multiple_regression(csv_file, target, features):
    df = pd.read_csv(csv_file)
    x = df[features]
    # separate the predicting attribute into Y for model training
    y = df[target]

    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)

    # creating an object of LinearRegression class
    LR = LinearRegression()
    # fitting the training data
    LR.fit(x_train, y_train)

    y_prediction = LR.predict(x_test)

    # Calculating the error that is not explained by the model
    score = r2_score(y_test, y_prediction)
    print("Accuracy of the model is:", score)
    print('Mean Squared Error:', mean_squared_error(y_test, y_prediction))


print("2020 Data with Covid Cases")
# Model with Covid cases taken into account
multiple_regression('sDeaths2020_count.csv', 'Total_deaths',
                    ["Age", "Race", "Gender", "Total_cases"])

print("2020 Data without Covid Cases")
# Model without Covid cases taked into account
multiple_regression('sDeaths2020_count.csv', 'Total_deaths',
                    ["Age", "Race", "Gender"])

print("2010 Data without Covid Cases")
# Model without Covid cases taked into account
multiple_regression('sDeaths2019_count.csv', 'Total_deaths',
                    ["Age", "Race", "Gender"])

