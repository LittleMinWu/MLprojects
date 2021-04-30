# pictures' meanings
1. ridge_poly.png  
Assume K=10 in KFold, search the best polynomial features' degree in linear regression model  
conclusion: choosing degree=4  
poly_range: [2, 3, 4, 5, 6]
2. ridge_kfold.png  
Now using default poly degree, we are going to find the best K in KFold based on that.  
conclusion: Even if when K=10 the score is not the best, but there is an over-fitting trend after k=10, so made a trade-off and choose KFold as 8  
K_range: [2, 4, 6, 8, 10, 20, 40]
3. ridge_C.png
conclusion: actually this model's performance is very bad, we guess ridge is very sensitive to the amount of features,  
but according to the results anyway, we decided to selected the best C as default value since there is no big difference.  
C_range: [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]  
4. ridge_iteration.png  
conclusion: There is no big difference between different max iterations, so we choose default value as the best max_iteration  
max_iteration range: [1e3, 1e4, 1e5, 1e6]