# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

#LOSS FUNCTION
def calculate_loss_function(actual, pred):
    rootmeansquareerror = np.sqrt(mean_squared_error(actual, pred))
    return rootmeansquareerror

class LogesticRegressionModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.clf = LogisticRegression()


    def train_model(self, folds=1):
        
        kf = KFold(n_splits = folds)
        
        iterator = kf.split(self.X_train)
        
        accuracy_arr = []
        loss_arr = []
        for i in range(folds):
            train_index, valid_index = next(iterator)
            
            X_train, y_train = self.X_train.iloc[train_index], self.y_train.iloc[train_index]
            X_valid, y_valid = self.X_train.iloc[valid_index], self.y_train.iloc[valid_index]
                        
            self.clf = self.clf.fit(X_train, y_train)
            
            vali_pred = self.clf.predict(X_valid)
            
            accuracy = self.calculate_score(y_valid, vali_pred)
            loss = calculate_loss_function(y_valid, vali_pred)
            
            self.__printAccuracy(accuracy, i, label="Validation")
            self.__printLoss(loss, i, label="Validation")
            print()
            
            accuracy_arr.append(accuracy)
            loss_arr.append(loss)

            
        return self.clf, accuracy_arr, loss_arr
    
    
    def test_model(self):
        y_pred = self.clf.predict(self.X_test)
        
        accuracy = self.calculate_score(self.y_test, y_pred)
        self.__printAccuracy(accuracy, label="Test")
        
        report = self.report(y_pred, self.y_test)
        matrix = self.confusion_matrix(y_pred, self.y_test)
        loss = calculate_loss_function(self.y_test, y_pred)

        
        return accuracy, loss, report, matrix 
    
    
    def calculate_score(self, pred, actual):
        return metrics.accuracy_score(actual, pred)
    
    def __printLoss(self, loss, step=1, label=""):
        print(f"step {step}: {label} Loss of LogesticRegression: {loss:.3f}")
        
    def __printAccuracy(self, acc, step=1, label=""):
        print(f"step {step}: {label} Accuracy of LogesticRegression: {acc:.3f}")
        
    
    def report(self, pred, actual):
        print("Test Metrics")
        print("================")
        print(metrics.classification_report(pred, actual))
        return metrics.classification_report(pred, actual)
    
    def confusion_matrix(self, pred, actual):
        ax=sns.heatmap(pd.DataFrame(metrics.confusion_matrix(pred, actual)))
        plt.title('Confusion matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        return metrics.confusion_matrix(pred, actual)
    
    def calculate_p_values(self):
       
      # X = 
       d = (2.0*(1.0+np.cosh(self.clf.decision_function(X))))
       d = np.tile(d,(X.shape[1],1)).T
       F_im = np.dot((X/denom).T,X) ## Fisher Information Matrix
       Cramer_Rao = np.linalg.inv(F_im) ## Inverse Information Matrix
       sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
       z_scores = self.clf.coef_[0]/sigma_estimates # z-score 
       p_values = [stat.norm.sf(abs(x)) for x in z_scores] ### two tailed test for p-values
        
       p_value_df = pd.DataFrame()
       p_value_df['features'] = self.X_train.columns.to_list()
       p_value_df['p_values'] = p_values
        
       return p_value_df
    
    def plot_pvalues(self, p_value_df):
        
        fig, ax = plt.subplots(figsize=(12,7))

        ax.plot([0.05,0.05], [0.05,5])
        sns.scatterplot(data=p_value_df, y='features', x='p_values', color="green")
        plt.title("P values of features", size=20)

        plt.xticks(np.arange(0,max(p_value_df['p_values']) + 0.05, 0.05))

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.show()