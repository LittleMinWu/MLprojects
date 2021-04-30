import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from funcfile import *
df = pd.read_csv("week4_2.csv",comment='#')
#df = pd.read_csv("week4_2.csv")
print(df.head())
x = np.array(df.iloc[:, 0:2])#read the colunms as array
x1 = np.array(df.iloc[:, 0])#read the first colunm as array
x2 = np.array(df.iloc[:, 1])#read the second colunm as array
y = np.array(df.iloc[:, 2])#read the y value

fig1=plt.figure()
scattersth(x1,x2,y,"orignal data for dataset 1")

#plt.show()

#to plot the impact of different values of degree
#model:linear_model.LogisticRegression,penalty:L2,C=1
#use K-Fold(K=5) cross-validation 
#plot the mse and the mean scores of cross-validation
fig2=plt.figure()
polynomial_degrees=[2,3,4,5,6,10,12,15,40,50,60,80,100]
degree_mean_mse = []
degree_std_mse = []
mean_score_degree_range=[]
for inx1,degree in enumerate(polynomial_degrees):
    poly = PolynomialFeatures(degree=degree, interaction_only=False,include_bias=False)
    x_poly = poly.fit_transform(x)
    kf = KFold(n_splits=5)
    mse = []
    for train, test in kf.split(x_poly):
        mul_lr = linear_model.LogisticRegression(penalty='l2',C=1)
        mul_lr.fit(x_poly, y)
        y_pre = mul_lr.predict(x_poly[test])
        from sklearn.metrics import mean_squared_error
        mse.append(mean_squared_error(y_pre, y[test]))
    degree_mean_mse.append(np.mean(mse))
    degree_std_mse.append(np.std(mse))
    mean_score_degree_range.append(cross_val_score(mul_lr, x_poly, y, cv=5).mean())
plt.subplot(1,2,1)
plt.plot(polynomial_degrees,mean_score_degree_range,label='mean scores',color='red')
plt.ylabel('mean score')
plt.xlabel('polynomial degree')
plt.title("score of different polynomial degrees")
plt.legend(loc='upper right', fontsize=10)
plt.subplot(1,2,2)
plt.errorbar(polynomial_degrees, degree_mean_mse, label="MSE of cross_validation", color='blue', yerr=degree_std_mse)
plt.ylabel('Mean square error')
plt.xlabel('polynomial degree')
plt.title("MSE of different polynomial degrees")
plt.legend(loc='upper right', fontsize=10)
#plt.show()
#to plot the impact of different values of C
#model:linear_model.LogisticRegression,penalty:L2,degree=50
#use K-Fold(K=5) cross-validation 
#plot the mse and the mean scores of cross-validation
fig3=plt.figure()
crange_mean_mse = []
crange_std_mse = []
mean_score_c_range=[]
#c_range=[0.1,0.5,1,2,3,5,10,15,20,30,40,50,60]
c_range=[0.1,0.5,1,2,3,4,5,10]
for c in c_range:
    poly = PolynomialFeatures(degree=50, interaction_only=False, include_bias=False)
    x_poly = poly.fit_transform(x)
    kf = KFold(n_splits=5)
    mse = []
    for train, test in kf.split(x_poly):
        mul_lr = linear_model.LogisticRegression(penalty='l2', C=c)
        mul_lr.fit(x_poly, y)
        y_pre = mul_lr.predict(x_poly[test])
        from sklearn.metrics import mean_squared_error
        mse.append(mean_squared_error(y_pre, y[test]))
    crange_mean_mse.append(np.mean(mse))
    crange_std_mse.append(np.std(mse))
    mean_score_c_range.append(cross_val_score(mul_lr, x_poly, y, cv=5).mean())
plt.plot(c_range,mean_score_c_range,label='mean scores',color='red')
plt.errorbar(c_range, crange_mean_mse, label="MSE", color='blue', yerr=crange_std_mse)
plt.ylabel('Mean square error')
plt.xlabel('C')
plt.title("cross validation of different C values")
plt.legend(loc='upper right', fontsize=10)

#plot final choice
#(i)degree=4 C=1 penalty=L2
#(ii)degree=50 and C=1 ,penalty=L2
fig4=plt.figure()
poly_final = PolynomialFeatures(degree=50, interaction_only=False, include_bias=False)
x_poly_final = poly_final.fit_transform(x)
kf = KFold(n_splits=5)
mul_lr_final = linear_model.LogisticRegression(penalty='l2', C=1)
mul_lr_final.fit(x_poly_final, y)
y_pre_final=mul_lr_final.predict(x_poly_final)
print("The score of the model is {0}".format(mul_lr_final.score(x_poly_final,y)))
scattersth(x1,x2,y_pre_final,"prediction model:degree=50,C=1 for dataset 2")
plt.legend(loc='upper right', fontsize=10)

#(b) for KNN
#to plot the impact of different values of K for KNN model
#use K-Fold(K=5) cross-validation
#plot the mse and the mean scores of cross-validation
fig5=plt.figure()
knn_mean_mse = []
knn_std_mse = []
mean_score_k_range=[]
k_range=[2,3,4,5,6,10,15,20,30,40,50,60,70,80,90,100,200]
for k in k_range:
    kf = KFold(n_splits=5)
    mse = []
    for train, test in kf.split(x):
        from sklearn.neighbors import KNeighborsClassifier
        knn_model = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(x, y)
        y_pre = knn_model.predict(x[test])
        from sklearn.metrics import mean_squared_error
        mse.append(mean_squared_error(y_pre, y[test]))
    knn_mean_mse.append(np.mean(mse))
    knn_std_mse.append(np.std(mse))
    mean_score_k_range.append(cross_val_score(knn_model, x, y, cv=5).mean())
plt.plot(k_range,mean_score_k_range,label='mean scores',color='red')
plt.errorbar(k_range, knn_mean_mse, label="MSE", color='blue', yerr=knn_std_mse)
plt.ylabel('Mean square error')
plt.xlabel('K')
plt.title("cross validation of different K values")
plt.legend(loc='upper right', fontsize=10)

#to find the degree of polynomial
fig6=plt.figure()
polynomial_degrees=[2,3,4,5,6,10,12,15,20]
mean_score_degree_range=[]
for degree in polynomial_degrees:
    poly = PolynomialFeatures(degree=degree, interaction_only=False,include_bias=False)
    x_poly = poly.fit_transform(x)
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform').fit(x_poly, y)
    mean_score_degree_range.append(cross_val_score(knn_model, x_poly, y, cv=5).mean())
#this is knn model without poly features mean score of cross validation
knn_model_no_poly = KNeighborsClassifier(n_neighbors=5, weights='uniform').fit(x,y)
knn_no_poly_mean_score= cross_val_score(knn_model, x, y, cv=5).mean()
plt.axhline(y=knn_no_poly_mean_score, color='blue', label='no poly features score',linestyle='-')
plt.plot(polynomial_degrees,mean_score_degree_range,label='mean scores of different degree',color='red')
plt.ylabel('mean score')
plt.xlabel('polynomial degree')
plt.title("score of different polynomial degrees")
plt.legend(loc='lower right', fontsize=10)
#(c)confusion matrices
#for trained Logistic Regression and kNNclassifier.
#baseline:dummy classifier

##final choice of logistic regression
#(i)degree=4 C=1 penalty=L2
#(ii)degree=50 and C=1 ,penalty=L2
fig7=plt.figure()
poly_final = PolynomialFeatures(degree=50, interaction_only=False, include_bias=False)
x_poly_final = poly_final.fit_transform(x)
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(x_poly_final, y, random_state=0)
mul_lr_final = linear_model.LogisticRegression(penalty='l2', C=1)
mul_lr_final.fit(X_train_logistic, y_train_logistic)
y_pre_logistic=mul_lr_final.predict(X_test_logistic)
disp1=plot_confusion_matrix(mul_lr_final, X_test_logistic, y_test_logistic)
print("Logistic Regression confusion matrix:")
print(disp1.confusion_matrix)
print("classification report")
print(classification_report(y_test_logistic, y_pre_logistic))
plt.title("trained logistic regression")
#KNN model (i)K=5 (i)K=30
fig8=plt.figure()
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(x, y, random_state=0)
knn_model = KNeighborsClassifier(n_neighbors=30, weights='uniform').fit(X_train_knn,y_train_knn)
y_pre_knn = knn_model.predict(X_test_knn)
disp2=plot_confusion_matrix(knn_model,X_test_knn,y_test_knn)
print("KNN confusion matrix:")
print(disp2.confusion_matrix)
print("classification report")
print(classification_report(y_test_knn, y_pre_knn))
plt.title("KNN")

#baseline model dummy classifier
fig9=plt.figure()
from sklearn.dummy import DummyClassifier
X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(x, y, random_state=0)
dummy = DummyClassifier(strategy="most_frequent").fit(X_train_dummy, y_train_dummy)
ydummy = dummy.predict(X_test_dummy)
disp3=plot_confusion_matrix(dummy,X_test_dummy,y_test_dummy)
print("dummy classifier confusion matrix：")
print(disp3.confusion_matrix)
print("classification report")
print(classification_report(y_test_dummy, ydummy))
plt.title("dummy classifier")

#d ROC curve
fig10=plt.figure()
from sklearn.metrics import roc_curve
#logistic regression
fpr1, tp1, _ = roc_curve(y_test_logistic,mul_lr_final.decision_function(X_test_logistic))
plt.plot(fpr1,tp1,label='logistic regression',c='b',linewidth=6)
#KNN
y_scores_knn = knn_model.predict_proba(X_test_knn)
fpr2, tp2, threshold1 = roc_curve(y_test_knn, y_scores_knn[:, 1])
plt.plot(fpr2,tp2,label='KNN',c='y')
#baseline
y_scores_dummy = dummy.predict_proba(X_test_dummy)
fpr3, tp3, threshold2 = roc_curve(y_test_dummy, y_scores_dummy[:, 1])
plt.plot(fpr3,tp3,label='baseline model:dummy classifier',c='r',linewidth=6)

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot([0,1], [0,1], color="green",linestyle="--")
plt.title("ROC curve for logistic regression")
plt.legend()
plt.show()

