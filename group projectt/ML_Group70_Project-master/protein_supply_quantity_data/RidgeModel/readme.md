# pictures' meanings
1. ridge_poly.png  
Assume K=10 in KFold, search the best polynomial features' degree in linear regression model  
conclusion: choosing degree=4  
poly_range: [2, 3, 4, 5, 6]
2. ridge_kfold.png  
Now we get the best polynomial features' degree as 3, then we are going to find the best K in KFold based on that.  
conclusion: Even if when K=10 the score is not the best, but there is an over-fitting trend after k=10, so made a trade-off and choose KFold as 8  
K_range: [2, 4, 6, 8, 10, 20, 40]
