# pictures' meanings
1. forrest_poly.png  
Using default K in KFold as metrics to find the best poly degree.  
conclusion: when poly =2 then error is the smallest and the score is the biggest,  
so we choose 2 as the best polynomial features' degree  
degree range:
2. forrest_k.png
Now using default poly degree to search the best K in KFold  
conclusion: When k=6 the error reached the lowest, but when k=8, the score comes to near 0.9,  
which is a significant improvement, so we choose 8 as the best K  
k range:  [2, 4, 6, 8, 10, 20, 40]
3. forrest_criterion.png  
There are two optional criterion `"mae"` and `"mse"`  
conclusion:`mse` has better performance and the MSE error did not change obviously. So we choose `mse`  
as the best criterion  
criterion range: ['mae','mse']  
4. forrest_trees.png  
Searching the best `n_estimators`
conclusion: when the test error and the train error is smallest, the number of trees = 8, but  
when the score reached the biggest, there is no big gap in two kinds of error, so made a trade-off,we choose tress=20  
as the best tress.
trees range: [2, 4, 6, 8, 10, 20, 40, 60, 100]