import matplotlib.pyplot as plt
import sympy as sp

#plot the data
from matplotlib.ticker import MultipleLocator

"""when y’s value equals 1 color is “red” ,else “green”, and set the x axis as ‘x_1’ ,
set the y axis as ‘x_2’ , because on the plot the x-axis is the value of parameter ‘x1 ’,
the y-axis is the value of the second parameter ‘x1’ , and the parameter of ‘str’ is 
the title of the plot"""
def scattersth(x1, x2, y, str):
    positivey_x1=[]
    positivey_x2 = []
    negaivey_x1=[]
    negaivey_x2 = []
    for i in range(len(x1)):
        if y[i] == 1:
            positivey_x1.append(x1[i])
            positivey_x2.append(x2[i])
        else:
            negaivey_x1.append(x1[i])
            negaivey_x2.append(x2[i])
    plt.scatter(positivey_x1,positivey_x2, label='y=1',color='red', s=5)
    plt.scatter(negaivey_x1, negaivey_x2, label='y=-1',color='green', s=5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(str)
