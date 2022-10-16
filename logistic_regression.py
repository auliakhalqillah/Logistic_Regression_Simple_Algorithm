# update
# sources : 
# https://github.com/sawarni99/Simple-Logistic-Regression
# https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
# https://medium.com/analytics-vidhya/derivative-of-log-loss-function-for-logistic-regression-9b832f025c2d

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class LogisticRegressionManual:
    def __init__(self,alpha=1,iter=1000):
        self.alpha = alpha
        self.iter = iter

    def sigmoid(self,Z):
        s = 1/(1 + np.exp(-1*Z))
        return s

    def logfunc(self,beta,X):
        '''
        Sigmoid Function or Logistic Function in Logistic Regression
        
        Input:

        X = feature(s) - MxN dimensions
        
        beta = weight coefficient - N dimensions

        Output:

        prob = Prediction Probability
        
        z = Logistic Regression
        '''
        z = np.dot(X,beta.T)
        # z = np.array(sorted(z))
        prob = self.sigmoid(z)
        return prob,z

    # Cost Function
    def costFunction(self,y,prob):
        '''
        Cost Function is an error calculation for logistic regression

        Input:

        y = target/label - M dimensions
        
        prob = Prediction Probability

        Output:
        
        cost = Cost value
        '''
        y = np.asarray(y).reshape(1,len(y)).T
        class_cost1 = -y*(np.log(prob))
        class_cost2 = (1-y)*(np.log(1-prob))
        cost = class_cost1 - class_cost2
        return np.mean(cost)

    def updateWeight(self,X,y,prob,beta):
        '''
        Update weight in gradient descent

        Input:

        X = feature(s) - MxN dimensions
        
        y = target/label - M dimensions
        
        prob = prediction probability from logistic function
        
        beta = weight coefficient - N dimensions
        '''
        y = np.asarray(y).reshape(len(y),1)
        self.gradient = np.dot(X.T, (prob - y))
        self.gradient /=  len(X)
        self.gradient *=  self.alpha
        beta = beta.T - self.gradient
        return beta.T

    def fit_(self,X,y):
        '''
        To get a proper weight coefficient for logistic regression

        Input:

        X = feature(s) - MxN dimensions
        
        y = target/label M dimensions

        Output:

        beta = a proper weight coefficient - N dimensions
        '''
        # initialization
        onesValue = np.ones((X.shape[0],1))
        X = np.asarray(X)
        X = np.concatenate([onesValue,X], axis=1)

        m = X.shape[0]
        n = X.shape[1]
        beta = np.zeros((1,n))

        self.costValue = []
        for i in range(self.iter):
            # calculate hypothesis value
            predictions_prob,z = self.logfunc(beta, X)

            # Calculate cost value
            costVal = self.costFunction(y,predictions_prob)
            self.costValue.append(costVal)
            # self.J.append(J0)

            # update self.beta value
            beta = self.updateWeight(X,y,predictions_prob,beta)
        self.betaCoeff = beta
    
    def beta_(self):
        print('>> Beta Coefficient:', self.betaCoeff[0][1:])
        print('>> Beta Intercept:', self.betaCoeff[0][0])
    
    def predict_(self,X):
        '''
        Predict function for logistic regression

        Input:

        X = feature(s) - MxN dimensions

        Output:

        predValue = target/label predictions - M dimensions
        '''

        # Initialization ones series and ada to the firts column database
        onesValue = np.ones((X.shape[0],1))
        X = np.asarray(X)
        X = np.concatenate([onesValue,X], axis=1)

        # calculating logistic function
        self.predProb,self.z = self.logfunc(self.betaCoeff,X)
        # print(self.predProb)
        predValue = np.where(self.predProb >= .5, 1, 0)
        self.predValue = [p.tolist()[0] for p in predValue]
        return self.predValue, self.predProb
    
    def accuracy(self, predicted_labels, actual_labels):
        num = 0
        for i,actual in enumerate(actual_labels):
                if actual == predicted_labels[i]:
                    num += 1
        acc = (num/len(actual_labels)) * 100
        return acc

    def plot_cost_value(self):
        fig, ax = plt.subplots()
        iteration = np.arange(0,self.iter)
        ax.plot(iteration,self.costValue)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('$Cost Value; Cost(h_{{\u03B8}}(x),y)$')
        ax.set_title('Cost Faunction')
        plt.show()
    
    def plot_logfunc(self):
        zz = np.array(sorted(self.z))
        prob = self.sigmoid(zz)
        fig, ax = plt.subplots()
        ax.scatter(self.z,self.predValue, label='Prediction')
        ax.plot(zz,prob,'red', label='Logistic Fit')
        ax.set_xlabel('z')
        ax.set_ylabel('Probability')
        ax.set_title('Logistic Faunction')
        plt.legend()
        plt.show()
        return

# Read data set
dataset = pd.read_csv("networks.csv")
print('>> Data Set')
print(dataset.head())
print('>> Info Data Set')
print(dataset.info())

# Change the category object field to the category numeric field (Label Encoder)
encode = LabelEncoder()
scaler = StandardScaler()
gender_encoder = encode.fit_transform(dataset['Gender'])
age_scaler = scaler.fit_transform(np.asarray(dataset['Age']).reshape(-1,1), np.asarray(dataset['Purchased']).reshape(-1,1))
salary_scaler = scaler.fit_transform(np.asarray(dataset['EstimatedSalary']).reshape(-1,1), np.asarray(dataset['Purchased']).reshape(-1,1))
dataset.insert(3,'Gender-Encode',gender_encoder)
dataset.insert(4,'Age-scaler',age_scaler)
dataset.insert(5,'EstimatedSalary-scaler',salary_scaler)
print('>> After Encoder')
print(dataset.head())

# Select the main data set
X = dataset[['Gender-Encode','Age-scaler', 'EstimatedSalary-scaler']]
# Select the target data set
y = dataset['Purchased'].tolist()

# split the data into two parts (training and testing data set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(
'''
>> Split the data
   - X_train: %d
   - X_test: %d
   - y_train: %d
   - y_test: %d
'''
% (len(X_train) ,len(X_test),len(y_train), len(y_test))
    )

# Building Logistic regression Manual
logreg = LogisticRegressionManual(iter=100)
# Initializations
print('>> Building Logistic regression')
m = X_train.shape[0]
n = X_train.shape[1]

# fit_ the data
logreg.fit_(X_train,y_train)

# Print beta coefficient
logreg.beta_()

# Plot cost function for each iteration
logreg.plot_cost_value()

# predict_
y_pred, y_prob = logreg.predict_(X_test)

# plot logistic function
logreg.plot_logfunc()

logreg.accuracy(y_pred, y_test)
accuracy_value = logreg.accuracy(y_pred,y_test)
print(' - Accuracy: %.2f percent' % accuracy_value)

tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print(' - True Negative: %d' % tn)
print(' - False Positive: %d' % fp)
print(' - False Negative: %d' % fn )
print(' - True Positive: %d' % tp)
print(' - Classification Report:')
print('   ', classification_report(y_test, y_pred))


# Logistic Regression Scikit Learn
print('\n>> Building Logistic regression Scikit Learn')
model = LogisticRegression()
model.fit(X_train,y_train)
print('>> Beta Coefficient:', model.coef_)
print('>> Intercept:', model.intercept_)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)*100
print(' - Accuracy: %.2f percent' % acc)
tn, fp, fn, tp = confusion_matrix(y_test,predictions).ravel()
print(' - True Negative: %d' % tn)
print(' - False Positive: %d' % fp)
print(' - False Negative: %d' % fn )
print(' - True Positive: %d' % tp)
print(' - Classification Report:')
print('   ', classification_report(y_test, predictions))
