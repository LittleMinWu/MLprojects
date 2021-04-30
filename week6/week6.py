import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

from gaussian_kernel_funcfile import *
#(i)(a)
x_dummy_training_data=np.array([-1,0,1]).reshape(-1,1)
y_dummy_training_data=np.array([0,1,0]).reshape(-1,1)
fig1=plt.figure()
from sklearn.neighbors import KNeighborsRegressor
# KNeighborsRegressor model :k = 3  γ=0, 1, 5, 10, 25
# generate predictions on a grid of feature values that range from -3 to 3
Xtest =[]
grid=np.linspace(-3,3)
for x in grid:
    Xtest.append(x)
Xtest=np.array(Xtest).reshape(-1,1)
model1 = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel0).fit(x_dummy_training_data, y_dummy_training_data)
ypred1 = model1.predict(Xtest)
model2 = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel1).fit(x_dummy_training_data, y_dummy_training_data)
ypred2 = model2.predict(Xtest)
model3 = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel5).fit(x_dummy_training_data, y_dummy_training_data)
ypred3 = model3.predict(Xtest)
model4 = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel10).fit(x_dummy_training_data, y_dummy_training_data)
ypred4 = model4.predict(Xtest)
model5 = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel25).fit(x_dummy_training_data, y_dummy_training_data)
ypred5 = model5.predict(Xtest)
plt.scatter(x_dummy_training_data, y_dummy_training_data, color='red',label="dummy_training_data")#plot the dummy training data
plt.plot(Xtest,ypred1,color='green',label="k=3,gamma=0")
plt.plot(Xtest, ypred2, color='blue',label="k=3,gamma=1")
plt.plot(Xtest, ypred3, color='orange',label="k=3,gamma=5")
plt.plot(Xtest, ypred4, color='purple',label="k=3,gamma=10")
plt.plot(Xtest, ypred5, color='brown',label="k=3,gamma=25")
plt.xlabel('input x')
plt.ylabel('output y')
plt.title("predictions of training data")
plt.legend()
plt.show()
#(b)
fig2=plt.figure()
Xtest_1 =[]
grid=np.linspace(-1,1)
for x in grid:
    Xtest_1.append(x)
Xtest_1=np.array(Xtest_1).reshape(-1,1)
ypred_1=model5.predict(Xtest_1)
plt.scatter(x_dummy_training_data, y_dummy_training_data, color='red',label="dummy_training_data")#plot the dummy training data
plt.plot(Xtest_1, ypred_1, color='blue',label="k=3,gamma=25")
plt.xlabel('input x')
plt.ylabel('output y')
plt.legend()
plt.show()
#(c)
from sklearn.kernel_ridge import KernelRidge
c_range=[0.1,1,1000]
gamma_range=[0,1,5,10,25]
color_range=['blue','green','orange','lime','cyan','pink','purple','cornflowerblue']
"""
for inx1,gamma in enumerate(gamma_range):
    fig3 = plt.figure() #for each gamma has one plot with different values of C
    plt.scatter(x_dummy_training_data, y_dummy_training_data, color='red',
                label="dummy_training_data",s=10)  # plot the dummy training data
    for inx2,C in enumerate(c_range):
        model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=gamma).fit(x_dummy_training_data, y_dummy_training_data)
        ypred_kernelridge=model.predict(Xtest)
        print("KernelRidge model when gammma={0},C={1},dual_coef={2}".format(gamma,C,model.dual_coef_))
        plt.plot(Xtest,ypred_kernelridge,color=color_range[inx2],label='C={}'.format(C))
        plt.legend(loc='upper left', fontsize=10)
        plt.title('gamma={}'.format(gamma))
    plt.show()
"""
#(ii)(a)
gamma_range1=[0,1,5,10,25,50,100,200]
df = pd.read_csv("week6.csv")
x = np.array(df.iloc[:, 0]).reshape(-1,1)#read the x value as array
y = np.array(df.iloc[:, 1]).reshape(-1,1)#read the y value
# kNN model with Gaussian kernel weights: k=999 gamma=0,1,5,10,25
weights_range=[gaussian_kernel0,gaussian_kernel1,gaussian_kernel5,gaussian_kernel10,gaussian_kernel25,gaussian_kernel50,gaussian_kernel100,gaussian_kernel200]
fig4=plt.figure()
plt.scatter(x, y, color='red', label="original training data",s=3)  # plot the CSV training data
for inx3,weights in enumerate(weights_range):
    model_knn = KNeighborsRegressor(n_neighbors=999, weights=weights).fit(x, y)
    ypred_knn = model_knn.predict(Xtest)
    plt.plot(Xtest, ypred_knn, color=color_range[inx3],label='gamma={}'.format(gamma_range1[inx3]))
    plt.xlabel('input x')
    plt.ylabel('output y')
    plt.title('KNN model predictions when k=999 with different gamma')
    plt.legend(fontsize=5)
plt.show()
#(ii)(b)
"""
for inx4,gamma in enumerate(gamma_range1):
    fig5 = plt.figure() #for each gamma has one plot with different values of C
    plt.scatter(x, y, color='red',
                label="original training data",s=10)  # plot the original training data
    for inx4,C in enumerate(c_range):
        model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=gamma).fit(x, y)
        ypred_kernelridge=model.predict(Xtest)
        print("KernelRidge model when gammma={0},C={1},dual_coef={2}".format(gamma,C,model.dual_coef_))
        plt.plot(Xtest,ypred_kernelridge,color=color_range[inx4],label='C={}'.format(C))
        plt.legend(loc='upper left', fontsize=10)
        plt.title('gamma={}'.format(gamma))
    plt.show()"""
#(ii)(c)
""" Use cross-validation to choose a reasonable value for hyperparameter γ for the
kNN model. Now use cross-validation to choose γ and α hyperparameter for the
kernalised ridge regression model. Generate predictions for both models using
these “optimised” hyperparameter values. """
fig6=plt.figure()
mean_error_knn=[]
std_error_knn=[]
mean_score_knn=[]
weights_range1=[gaussian_kernel0,gaussian_kernel1,gaussian_kernel5,gaussian_kernel10,gaussian_kernel25]

for inx5,weights in enumerate(weights_range):
    kf = KFold(n_splits=5)
    model_knn1 = KNeighborsRegressor(n_neighbors=799, weights=weights)
    mse = []
    for train, test in kf.split(x):
        model_knn1.fit(x[train], y[train])
        y_pre_knn1 = model_knn1.predict(x[test])
        from sklearn.metrics import mean_squared_error
        mse.append(mean_squared_error(y[test],y_pre_knn1))
    mean_error_knn.append(np.array(mse).mean())
    std_error_knn.append(np.array(mse).std())
    mean_score_knn.append(cross_val_score(model_knn1, x, y, cv=5).mean())
plt.subplot(1,2,1)
plt.plot(gamma_range1,mean_score_knn,label='mean scores',color='red')
plt.ylabel('mean score')
plt.xlabel('gamma')
plt.title("score of different gamma values")
plt.legend(loc='upper right', fontsize=10)
plt.subplot(1,2,2)
plt.errorbar(gamma_range1, mean_error_knn, label="MSE of cross_validation", color='blue', yerr=std_error_knn)
plt.ylabel('Mean square error')
plt.xlabel('gamma')
plt.title("MSE of different gamma values")
plt.legend(loc='upper right', fontsize=10)
plt.show()

fig7=plt.figure()
model_knn_final = KNeighborsRegressor(n_neighbors=999, weights=gaussian_kernel50).fit(x,y)
ypred_knn_final=model_knn_final.predict(Xtest)
plt.scatter(x, y, color='red', label="original training data",s=3)  # plot the CSV training data
plt.plot(Xtest,ypred_knn_final,color='blue',label='prediction')
plt.title("KNN final model:K=999,gamma=50")
plt.legend()
#alpha=5 to find gamma for kernalised ridge regression model
mean_error_kr=[]
std_error_kr=[]
mean_score_kr=[]
for inx5,gamma in enumerate(gamma_range):
    kf = KFold(n_splits=5)
    kr_model = KernelRidge(alpha=5, kernel='rbf', gamma=gamma)
    mse = []
    for train, test in kf.split(x):
        kr_model.fit(x[train], y[train])
        y_pre_kf = kr_model.predict(x[test])
        from sklearn.metrics import mean_squared_error
        mse.append(mean_squared_error(y[test],y_pre_kf))
    mean_error_kr.append(np.array(mse).mean())
    std_error_kr.append(np.array(mse).std())
    mean_score_kr.append(cross_val_score(kr_model, x, y, cv=5).mean())
plt.subplot(1,2,1)
plt.plot(gamma_range,mean_score_kr,label='mean scores',color='red')
plt.ylabel('mean score')
plt.xlabel('gamma')
plt.title("score of different gamma values")
plt.legend(loc='upper right', fontsize=10)
plt.subplot(1,2,2)
plt.errorbar(gamma_range, mean_error_kr, label="MSE of cross_validation", color='blue', yerr=std_error_kr)
plt.ylabel('Mean square error')
plt.xlabel('gamma')
plt.title("MSE of different gamma values")
plt.legend(loc='upper right', fontsize=10)
#gamma=5 to find the best alpha
fig8=plt.figure()
mean_error_kr1=[]
std_error_kr1=[]
mean_score_kr1=[]
alpha_range=[0.01,0.03,0.05,0.07,0.1,0.15,0.2]
for alpha in alpha_range:
    kf = KFold(n_splits=5)
    kr_model1 = KernelRidge(alpha=alpha, kernel='rbf', gamma=5)
    mse = []
    for train, test in kf.split(x):
        kr_model1.fit(x[train], y[train])
        y_pre_kr = kr_model1.predict(x[test])
        from sklearn.metrics import mean_squared_error
        mse.append(mean_squared_error(y[test],y_pre_kr))
    mean_error_kr1.append(np.array(mse).mean())
    std_error_kr1.append(np.array(mse).std())
    mean_score_kr1.append(cross_val_score(kr_model1, x, y, cv=5).mean())
plt.subplot(1,2,1)
plt.plot(alpha_range,mean_score_kr1,label='mean scores',color='red')
plt.ylabel('mean score')
plt.xlabel('alpha')
plt.title("score of different alpha values")
plt.legend(loc='upper right', fontsize=10)
plt.subplot(1,2,2)
plt.errorbar(alpha_range, mean_error_kr1, label="MSE of cross_validation", color='blue', yerr=std_error_kr1)
plt.ylabel('Mean square error')
plt.xlabel('alpha')
plt.title("MSE of different alpha values")
plt.legend(loc='upper right', fontsize=10)
plt.show()
#plot final KR model gamma=5 alpha=0.1
fig9=plt.figure()
kr_model_final = KernelRidge(alpha=0.1, kernel='rbf', gamma=5).fit(x,y)
kr_pre_final=kr_model_final.predict(Xtest)
plt.scatter(x, y, color='red', label="original training data",s=3)  # plot the CSV training data
plt.plot(Xtest,kr_pre_final,color='blue',label='KernelRidge')
plt.plot(Xtest,ypred_knn_final,color='yellow',label='KNN')
plt.title("final models")
plt.legend()
plt.show()