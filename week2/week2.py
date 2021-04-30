import numpy as np
import pandas as pd
from funcfile import *
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#read data
df = pd.read_csv("week2.csv",comment="#")
print(df.head())
x = np.array(df.iloc[:, 0:2])#read the first two colunms as array
y = np.array(df.iloc[:, 4])#read the y value

#train the logistic regression classifier on the data
mul_lr = linear_model.LogisticRegression()
mul_lr.fit(x,y)
y_pre = mul_lr.predict(x)
print("linear_model.LogisticRegression :intercept {0}, slope {1} , score {2}".format(mul_lr.intercept_, mul_lr.coef_,mul_lr.score(x,y)))

x1 = x[:, 0]
x2 = x[:, 1]


plt.subplot(331)
scattersth(x1, x2, y, "original data")
plotsth(x1, mul_lr.intercept_, mul_lr.coef_)

plt.subplot(332)
plt.plot()
scattersthothercolor(x1, x2, y,y_pre, "logistic_regression prediction data")
plt.show()
#pick 6 numbers of C to train the model , and print the parameters, 
#after that predict the data then plot it , using the “add_subplot” function
fig2 = plt.figure()
svc_c = [0.001,1,10,100,500,1000]
for i in range(6):
    linear_svc_model = LinearSVC(C=svc_c[i]).fit(x, y)
    print("linear_svc_model{0} :when C={1},intercept {2}, slope {3} ,score {4}".format(i+1,svc_c[i], linear_svc_model.intercept_,
                                                                                          linear_svc_model.coef_, linear_svc_model.score(x,y)))
    y_svc_pre = linear_svc_model.predict(x)
    linear_svc_model_subplot = fig2.add_subplot(2, 3, i+1)
    scattersth(x1, x2, y_svc_pre, "predictiondata when c is {0}".format(svc_c[i]))
    plotsth(x1, linear_svc_model.intercept_, linear_svc_model.coef_)

plt.show()


# (c)
x_new = np.array(df.iloc[:, 0:4])
mul_lr1 = linear_model.LogisticRegression()
mul_lr1.fit(x_new, y)
print("new linear_model.LogisticRegression :intercept {0}, slope {1} ".format(mul_lr1.intercept_, mul_lr1.coef_))
y_pre1 = mul_lr1.predict(x_new)
plt.subplot(121)
scattersth(x1, x2, y_pre1, "new linear_model.LogisticRegression")
plt.axhline(y=0, color='yellow', linestyle='-' , label = 'baseline')
plt.legend()
plt.subplot(122)
scattersth(x1, x2, y, "original data")
plt.axhline(y=0, color='yellow', linestyle='-' , label = 'baseline')
plt.legend()
"""define each of the coefficient as a,b,c,d , and e is the intercept , then I apply the mathematic quadratic formula """
x1_new = np.linspace(-1, 1, 9999)#to get the axis of x
a = mul_lr1.coef_[0][0]
b = mul_lr1.coef_[0][1]
c = mul_lr1.coef_[0][2]
d = mul_lr1.coef_[0][3]
e = mul_lr1.intercept_
m = e + a*x1_new + c*x1_new*x1_new
x2_new = (-b+(b*b-4*d*m)**0.5)/(2*d)
plt.plot(x1_new, x2_new, label=' decision boundary', color='pink')
scattersth(x1, x2, y, "original data")
plt.legend()
plt.show()



