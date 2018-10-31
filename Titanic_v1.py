import Data_preprocessing_v1 as dpv1
import clf_model_v1 as clfv1
import prediction_v1 as prev1

data_train_new = dpv1.Preprocess()
# print("Logistic model:")
# clfv1.Logistic_model(data_train_new)
# print("SVM model:")
# clfv1.SVM_model(data_train_new)
# print("MLP model:")
clf = clfv1.MLP_model(data_train_new)
# print("Voting model:")
# clf = clfv1.Vote_model(data_train_new)
prev1.predict(clf)
