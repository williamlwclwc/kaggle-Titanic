from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
import numpy as np


def cross_val(data_train, cv_num, classifier):
    #cross-validation
    all_data = data_train.filter(regex = 'Survived|Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title|Family|Isalone|Person|Ticket_same|Ismother')
    x = all_data.values[:, 1:]  
    y = all_data.values[:, 0]  
    cv_result = model_selection.cross_val_score(classifier, x, y, cv = cv_num)
    print(cv_result, np.mean(cv_result))


def Logistic_model(data_train):
    #initialize classifier and fit the data
    train_df = data_train.filter(regex = 'Survived|Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title|Family|Isalone|Person|Ticket_same|Ismother')
    train_np = train_df.values
    y = train_np[:, 0] # data(x) start from the second column
    X = train_np[:, 1:] # data(y) label is the first column
    classifier = LogisticRegression()
    classifier.fit(X, y)
    cross_val(data_train, 10, classifier)


def SVM_model(data_train):
    #initialize classifier and fit the data
    train_df = data_train.filter(regex = 'Survived|Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title|Family|Isalone|Person|Ticket_same|Ismother')
    train_np = train_df.values
    y = train_np[:, 0] # data(x) start from the second column
    X = train_np[:, 1:] # data(y) label is the first column
    classifier = SVC(kernel = 'rbf', probability = True)
    classifier.fit(X, y)
    cross_val(data_train, 10, classifier)


def MLP_model(data_train):
    #initialize classifier and fit the data
    train_df = data_train.filter(regex = 'Survived|Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title|Family|Isalone|Person|Ticket_same|Ismother')
    train_np = train_df.values
    y = train_np[:, 0] # data(x) start from the second column
    X = train_np[:, 1:] # data(y) label is the first column
    classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 3), max_iter = 3000, 
                               random_state = 47, activation = 'relu')
    classifier.fit(X, y)
    cross_val(data_train, 10, classifier)
    return classifier


def Vote_model(data_train):
    train_df = data_train.filter(regex = 'Survived|Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title|Family|Isalone|Person|Ticket_same|Ismother')
    train_np = train_df.values
    y = train_np[:, 0] # data(x) start from the second column
    X = train_np[:, 1:] # data(y) label is the first column

    # use vote strategy to merge models together
    log_clf = LogisticRegression()
    svm_clf = SVC(kernel = 'rbf', probability = True)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 100, 3), max_iter = 3000, 
                               random_state = 47, activation = 'relu')
    classifier = VotingClassifier(estimators = [('logistic',log_clf), ('SVC', svm_clf), ('DNN',mlp_clf)], voting = 'hard', n_jobs = -1)
    classifier.fit(X, y)
    # cross_val(data_train, 10, classifier)
    return classifier