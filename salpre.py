import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import dataset
df=pd.read_csv("Salary_data.csv")
X=df.iloc[:, :-1]
y=df.iloc[:, 1]
#spliting dataset into two part trainingset & test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train ,y_test =train_test_split(X, y, test_size=1/3 ,random_state=0)
#train the model using linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#predict the testset result
y_pred=regressor.predict(X_test)
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
