import matplotlib.pyplot as plt
import sympy as sp

#plot the data
"""when y’s value equals 1 color is “red” ,else “green”, and set the x axis as ‘x_1’ ,
set the y axis as ‘x_2’ , because on the plot the x-axis is the value of parameter ‘x1 ’,
the y-axis is the value of the second parameter ‘x1’ , and the parameter of ‘str’ is 
the title of the plot"""
def scattersth(x1, x2, y, str):
    for i in range(len(x1)):
        if y[i] == 1:
            plt.scatter(x1[i], x2[i], color='red', s=5)
        else:
            plt.scatter(x1[i], x2[i], color='green', s=5)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title(str)

""" plot the two datasets and could clearly see the different y value of the original ones and the prediction ones"""
def scattersthothercolor(x1, x2, y,y_pre , str):
    for i in range(len(x1)):
        if y[i] == 1 :
            plt.scatter(x1[i], x2[i], color='red', s=5)
        if y_pre[i] == 1:
            plt.scatter(x1[i], x2[i], color='yellow', s=1)
        if y[i] == -1:
            plt.scatter(x1[i], x2[i], color='green', s=5)
        if y_pre[i] == -1 :
            plt.scatter(x1[i], x2[i], color='black', s=1)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title(str)

"""plot the line with intercept and coef"""
def plotsth(x, intercept, coef):
    plt.plot(x, -(intercept+coef[0][0]*x)/coef[0][1], color='blue', label='decision boundary')
    plt.legend()

