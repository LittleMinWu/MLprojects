
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read data
df = pd.read_csv("week1.csv")
print(df.head())

#function to have the Scaling factor
def deviation(array,mean):
    b=0
    sum=0
    for a in range(len(array)):
        b=(array[a]-mean)**2
        sum = b+sum
    return ((sum/len(array))**0.5)


x = np.array(df.iloc[:, 0])
x = x.reshape(-1, 1)
y = np.array(df.iloc[ :, 1])
y = y.reshape(-1, 1 )
#normalise x
mean_x = x.sum()/len(x)
deviation_x = deviation(x,mean_x)
normalised_x = (x - mean_x) / deviation_x
#normalise y
mean_y = y.sum()/len(y)
deviation_y = deviation(y,mean_y)
normalised_y = (y - mean_y) / deviation_y


#uses gradient descent to train a linear regression model
def costfunction(theta,b,x,y):
    a = np.power(((x*theta) +b - y),2) #square
    return np.sum(a)/len(x)


def costpei(weights,theta_0,x_array,y_array):
    temp = 0.
    for item in range(len(x_array)):
        temp += (theta_0 + weights * x_array[item] - y_array[item]) ** 2
    cost = temp / len(x_array)
    return cost


#define paramaters
#learning rate alpha
initial_alpha = 0.01
initial_theta=1
initial_b=1
iterations=999

def step_grad_desc(current_theta,current_b,alpha,x,y):
    sumgrad_theta=0
    sumgrad_b=0
    M=len(x)
    #every point in formula
    for i in range(M):
        sumgrad_theta += (current_theta * x[i] +current_b -y[i]) *x[i]
        sumgrad_b += (current_theta * x[i] +current_b - y[i])
    #用公式求当前梯度
    grad_theta=1/M * sumgrad_theta
    grad_b = 1/M * sumgrad_b
    #update theta and b
    updated_theta = current_theta - alpha*grad_theta
    updated_b = current_b - alpha*grad_b
    return updated_theta,updated_b

#define gradient descent function
def gradientdescent(x,y,theta,b,alpha,num_iter):
#define a list have all the loss function values to show the process of descent
    cost_list = []
    updated_theta = theta
    updated_b = b
    for i in range(num_iter):
        #initial loss func value
        cost_list.append(costfunction(updated_theta,updated_b,normalised_x,normalised_y))
        updated_theta, updated_b = step_grad_desc(updated_theta,updated_b,alpha,x,y)
    return [updated_theta , updated_b , cost_list]

"""
#test to have the optimal theta and b
theta, b , cost_list = gradientdescent(normalised_x,normalised_y,initial_theta,initial_b,initial_alpha,iterations)
cost=costfunction(initial_theta,initial_b,normalised_x,normalised_y)
print('theta is:', theta)
print('b is:', b)
print('cost is:', cost)


plt.plot(normalised_x, normalised_x*theta+b , color='red', label='model')
plt.scatter(normalised_x,normalised_y, s=1 ,color='blue', label='data')
plt.axhline(y=normalised_y.mean(), color='yellow', linestyle='-' , label = 'baseline')
plt.xlabel('normalised x')
plt.ylabel('normalised y')
plt.legend()
plt.show()
"""

"""draw learning rate
theta, b , cost_list1 = gradientdescent(normalised_x,normalised_y,initial_theta,initial_b,0.001,iterations)
theta, b , cost_list2 = gradientdescent(normalised_x,normalised_y,initial_theta,initial_b,0.01,iterations)
theta, b , cost_list3 = gradientdescent(normalised_x,normalised_y,initial_theta,initial_b,0.1,iterations)

plt.plot(cost_list1 , color = 'red',label = 'learning rate is 0.001')
plt.plot(cost_list2 , color = 'blue',label = 'learning rate is 0.01')
plt.plot(cost_list3 , color = 'green',label = 'learning rate is 0.1')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('cost')
plt.show()
"""

from sklearn.linear_model import LinearRegression

model =LinearRegression()
model.fit(normalised_x,normalised_y)
plt.plot(normalised_x, model.predict(normalised_x) ,color='yellow',linewidth=10,label='sklearn trained')

theta1, b1, cost_list1 = gradientdescent(normalised_x,normalised_y,initial_theta,initial_b,0.1,iterations)
theta2, b2, cost_list2 = gradientdescent(normalised_x,normalised_y,initial_theta,initial_b,0.01,iterations)
theta3, b3, cost_list3 = gradientdescent(normalised_x,normalised_y,initial_theta,initial_b,0.001,iterations)
plt.plot(normalised_x, normalised_x*theta1+b1 , color='red', linewidth=5,label='learningrate=0.1')
plt.plot(normalised_x, normalised_x*theta2+b2 , color='pink', label='learningrate=0.01')
plt.plot(normalised_x, normalised_x*theta3+b3 , color='green', label='learningrate=0.001')
plt.scatter(normalised_x,normalised_y, s=1 ,color='blue', label='data')

plt.xlabel('normalised x')
plt.ylabel('normalised y')
plt.legend()
plt.show()




