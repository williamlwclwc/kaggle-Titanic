import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2 

# load dataset
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

# data_train.info() # preview the data

# miss so much Cabin info
data_train.loc[(data_train.Cabin.notnull()), 'Cabin'] = 1
data_train.loc[(data_train.Cabin.isnull()), 'Cabin'] = 0

# calculate each title's average age
Mr_age_mean = (data_train[data_train.Name.str.contains('Mr.')]['Age'].mean())
Mrs_age_mean = (data_train[data_train.Name.str.contains('Mrs.')]['Age'].mean())
Miss_age_mean = (data_train[data_train.Name.str.contains('Miss.')]['Age'].mean())
Master_age_mean = (data_train[data_train.Name.str.contains('Master.')]['Age'].mean())
# Dr_age_mean = (data_train[data_train.Name.str.contains('Dr.')]['Age'].mean())
# use the average age to fill the missing age
data_train.loc[(data_train['Name'].str.contains('Mr.')) & data_train.Age.isnull(), 'Age'] = Mr_age_mean
data_train.loc[(data_train['Name'].str.contains('Mrs.')) & data_train.Age.isnull(), 'Age'] = Mrs_age_mean
data_train.loc[(data_train['Name'].str.contains('Miss.')) & data_train.Age.isnull(), 'Age'] = Miss_age_mean
data_train.loc[(data_train['Name'].str.contains('Master.')) & data_train.Age.isnull(), 'Age'] = Master_age_mean
data_train.loc[(data_train['Name'].str.contains('Dr.')) & data_train.Age.isnull(), 'Age'] = Mr_age_mean

# discretization all data
data_train.loc[data_train.Fare <= 7.91, 'Fare'] = 0
data_train.loc[(data_train.Fare > 7.91) & (data_train.Fare <= 14.454), 'Fare'] = 1
data_train.loc[(data_train.Fare > 14.454) & (data_train.Fare <= 31), 'Fare'] = 2
data_train.loc[data_train.Fare > 31, 'Fare'] = 3
data_train.loc[data_train.Age <= 16, 'Age'] = 0
data_train.loc[(data_train.Age > 16) & (data_train.Age < 32), 'Age'] = 1
data_train.loc[(data_train.Age > 32) & (data_train.Age < 48), 'Age'] = 2
data_train.loc[(data_train.Age > 48) & (data_train.Age < 64), 'Age'] = 3
data_train.loc[data_train.Age > 64, 'Age'] = 4
# importance is irrelavent to values: get dummy attributes
Age_dummies = pd.get_dummies(data_train['Age'], prefix = 'Age')
Fare_dummies = pd.get_dummies(data_train['Fare'], prefix = 'Fare')
Cabin_dummies = pd.get_dummies(data_train['Cabin'], prefix = 'Cabin')
Embarked_dummies = pd.get_dummies(data_train['Embarked'], prefix = 'Embarked')
Sex_dummies = pd.get_dummies(data_train['Sex'], prefix = 'Sex')
Pclass_dummies = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')
SibSp_dummies = pd.get_dummies(data_train['SibSp'], prefix = 'SibSp')
Parch_dummies = pd.get_dummies(data_train['Parch'], prefix = 'Parch')

# explore hidden attributes
# add attribute "title": first split the second part of string name, and then spilt the title out 
data_train['Title'] = data_train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
data_train['Title'] = pd.get_dummies(data_train['Title'], prefix = 'Title')
# add attribute "isalone": no sibsp & parch -> isalone
data_train['Isalone'] = np.nan
data_train.loc[data_train.SibSp + data_test.Parch == 0, 'Isalone'] = 1
data_train.loc[data_train.Isalone.isnull(), 'Isalone'] = 0 
# add attribute "ismother"
data_train['Ismother'] = np.nan
data_train.loc[(data_train.Parch > 0) & (data_train.Sex == 'female'), 'Ismother'] = 1
data_train.loc[(data_train.Ismother.isnull()), 'Ismother'] = 0
# add attribute "family"
data_train['Family'] = np.nan
data_train.loc[data_train.SibSp + data_train.Parch == 0, 'Family'] = 0 # no family
data_train.loc[(data_train.SibSp + data_train.Parch > 0) & (data_train.SibSp + data_train.Parch <= 3), 'Family'] = 1 # 1-3
data_train.loc[data_train.Family.isnull(), 'Family'] = 2 # too much people in one family
data_train['Family'] = pd.get_dummies(data_train['Family'])
# add attribute "person"
data_train['Person'] = np.nan
data_train.loc[data_train.Age <= 16, 'Person'] = 'child'
data_train.loc[(data_train.Age > 16) & (data_train.Sex == 'female'), 'Person'] = 'adult_woman'
data_train.loc[(data_train.Age > 16) & (data_train.Sex == 'male'), 'Person'] = 'adult_man'
data_train['Person'] = pd.get_dummies(data_train['Person'], prefix = 'Person')
# add attribute "ticket_same"
data_train['Ticket_same'] = np.nan
data_train['Ticket_same'] = data_train['Ticket'].duplicated()
data_train.loc[data_train.Ticket_same == True, 'Ticket_same'] = 1
data_train.loc[data_train.Ticket_same == False, 'Ticket_same'] = 0

# assess the importance of each attribute
data_train['Age'] = Age_dummies
data_train['Fare'] = Fare_dummies
data_train['Cabin'] = Cabin_dummies
data_train['Embarked'] = Embarked_dummies
data_train['Sex'] = Sex_dummies
data_train['Pclass'] = Pclass_dummies
data_train['SibSp'] = SibSp_dummies
data_train['Parch'] = Parch_dummies

predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "Sex",  
              "Title", "Family", "Isalone", "Ismother", "Person", "Ticket_same"]
selector = SelectKBest(score_func = chi2, k = 14)
a = selector.fit(data_train[predictors], data_train['Survived'])
print(np.array(a.scores_), '\n', a.get_support())
ax = sns.barplot(x = predictors, y = np.array(a.scores_), ci = 0)
plt.show()


