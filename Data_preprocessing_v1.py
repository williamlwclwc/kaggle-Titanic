import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2 


def csv_preprocessing(data):

    # data.info() # preview the data

    # miss so much Cabin info
    data.loc[(data.Cabin.notnull()), 'Cabin'] = 1
    data.loc[(data.Cabin.isnull()), 'Cabin'] = 0

    # calculate each title's average age
    Mr_age_mean = (data[data.Name.str.contains('Mr.')]['Age'].mean())
    Mrs_age_mean = (data[data.Name.str.contains('Mrs.')]['Age'].mean())
    Miss_age_mean = (data[data.Name.str.contains('Miss.')]['Age'].mean())
    Master_age_mean = (data[data.Name.str.contains('Master.')]['Age'].mean())
    # Dr_age_mean = (data[data.Name.str.contains('Dr.')]['Age'].mean())
    # use the average age to fill the missing age
    data.loc[(data['Name'].str.contains('Mr.')) & data.Age.isnull(), 'Age'] = Mr_age_mean
    data.loc[(data['Name'].str.contains('Mrs.')) & data.Age.isnull(), 'Age'] = Mrs_age_mean
    data.loc[(data['Name'].str.contains('Miss.')) & data.Age.isnull(), 'Age'] = Miss_age_mean
    data.loc[(data['Name'].str.contains('Master.')) & data.Age.isnull(), 'Age'] = Master_age_mean
    data.loc[(data['Name'].str.contains('Dr.')) & data.Age.isnull(), 'Age'] = Mr_age_mean

    # discretization all data
    data.loc[data.Fare <= 7.91, 'Fare'] = 0
    data.loc[(data.Fare > 7.91) & (data.Fare <= 14.454), 'Fare'] = 1
    data.loc[(data.Fare > 14.454) & (data.Fare <= 31), 'Fare'] = 2
    data.loc[data.Fare > 31, 'Fare'] = 3
    data.loc[data.Age <= 16, 'Age'] = 0
    data.loc[(data.Age > 16) & (data.Age < 32), 'Age'] = 1
    data.loc[(data.Age > 32) & (data.Age < 48), 'Age'] = 2
    data.loc[(data.Age > 48) & (data.Age < 64), 'Age'] = 3
    data.loc[data.Age > 64, 'Age'] = 4
    # importance is irrelavent to values: get dummy attributes
    Age_dummies = pd.get_dummies(data['Age'], prefix = 'Age')
    Fare_dummies = pd.get_dummies(data['Fare'], prefix = 'Fare')
    Cabin_dummies = pd.get_dummies(data['Cabin'], prefix = 'Cabin')
    Embarked_dummies = pd.get_dummies(data['Embarked'], prefix = 'Embarked')
    Sex_dummies = pd.get_dummies(data['Sex'], prefix = 'Sex')
    Pclass_dummies = pd.get_dummies(data['Pclass'], prefix = 'Pclass')
    SibSp_dummies = pd.get_dummies(data['SibSp'], prefix = 'SibSp')
    Parch_dummies = pd.get_dummies(data['Parch'], prefix = 'Parch')

    # explore hidden attributes
    # add attribute "title": first split the second part of string name, and then spilt the title out 
    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    data['Title'] = pd.get_dummies(data['Title'], prefix = 'Title')
    # add attribute "isalone": no sibsp & parch -> isalone
    data['Isalone'] = np.nan
    data.loc[data.SibSp + data.Parch == 0, 'Isalone'] = 1
    data.loc[data.Isalone.isnull(), 'Isalone'] = 0 
    # add attribute "ismother"
    data['Ismother'] = np.nan
    data.loc[(data.Parch > 0) & (data.Sex == 'female'), 'Ismother'] = 1
    data.loc[(data.Ismother.isnull()), 'Ismother'] = 0
    # add attribute "family"
    data['Family'] = np.nan
    data.loc[data.SibSp + data.Parch == 0, 'Family'] = 0 # no family
    data.loc[(data.SibSp + data.Parch > 0) & (data.SibSp + data.Parch <= 3), 'Family'] = 1 # 1-3
    data.loc[data.Family.isnull(), 'Family'] = 2 # too much people in one family
    data['Family'] = pd.get_dummies(data['Family'])
    # add attribute "person"
    data['Person'] = np.nan
    data.loc[data.Age <= 16, 'Person'] = 'child'
    data.loc[(data.Age > 16) & (data.Sex == 'female'), 'Person'] = 'adult_woman'
    data.loc[(data.Age > 16) & (data.Sex == 'male'), 'Person'] = 'adult_man'
    data['Person'] = pd.get_dummies(data['Person'], prefix = 'Person')
    # add attribute "ticket_same"
    data['Ticket_same'] = np.nan
    data['Ticket_same'] = data['Ticket'].duplicated()
    data.loc[data.Ticket_same == True, 'Ticket_same'] = 1
    data.loc[data.Ticket_same == False, 'Ticket_same'] = 0

    # assess the importance of each attribute
    data['Age'] = Age_dummies
    data['Fare'] = Fare_dummies
    data['Cabin'] = Cabin_dummies
    data['Embarked'] = Embarked_dummies
    data['Sex'] = Sex_dummies
    data['Pclass'] = Pclass_dummies
    data['SibSp'] = SibSp_dummies
    data['Parch'] = Parch_dummies

    return data


def Preprocess():
    # load dataset
    data_train = pd.read_csv("train.csv")
    data_train = csv_preprocessing(data_train)
    predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "Sex",  
                "Title", "Family", "Isalone", "Ismother", "Person", "Ticket_same"]
    selector = SelectKBest(score_func = chi2, k = 14)
    a = selector.fit(data_train[predictors], data_train['Survived'])
    # print(np.array(a.scores_), '\n', a.get_support())
    sns.barplot(x = predictors, y = np.array(a.scores_), ci = 0)
    # plt.show()
    return data_train

