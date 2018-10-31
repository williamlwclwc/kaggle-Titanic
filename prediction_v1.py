import pandas as pd
import numpy as np
from Data_preprocessing_v1 import csv_preprocessing

def predict(classifier):
    data_test = pd.read_csv("test.csv")
    data_test = csv_preprocessing(data_test)

    test_df = data_test.filter(regex = 'Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title|Family|Isalone|Person|Ticket_same|Ismother')
    yhat = classifier.predict(test_df)
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'].values,
                            'Survived': yhat.astype(np.int32)})
    result.to_csv("result_v1.csv", index = False)
