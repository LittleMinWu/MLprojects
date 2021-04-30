## 1. All the hyper parameters which need be selected manually in each model
- Linear Regression Model
1. Degree of Polynomial features
2. K in KFold
- Ridge Model
1. C
2. Max Iteration
3. degree of polynomial features
4. K in KFold
- Random Forrest Regressor 
1. Number of Trees
2. criterion
3. degree of polynomial features
4. K in KFold
## 2. Data Preprocessing
In the step of data preprocessing, we used `pandas` to implement related functions.
```python
# fill N/A
# give a constant when there is a range of value
def get_features(file):
    obesity = file['Obesity']
    features = file.iloc[:, 1:24]
    # print(len(obesity.dropna())/len(obesity)) 98%, we do not need to drop those features with N/A
    obesity.fillna(obesity.mean(), inplace=True)
    return features, obesity
```
## 3. Project Target
We collected three data sets, which represented three different dimensions to describe multiple countries' people's   
daily food(fat proportion, food kcal proportion and protein proportion), we wanted to find a relatively general pattern,  
which could show which dimension will contribute the most to people's obesity, so that we can create a reference when we try to keep our own health.
## 4. Best Hyper parameters in three models
1. Food Supply KCAL
    1. Linear Regression Model
        1. K: 8
        2. Poly: 3
    2. Ridge Model
        1. C: default value
        2. max_iteration: default value
        3. K:8
        4. Poly:2
    3. Random Forrest Regression Model
        1. criterion: 'MSE'
        2. K: 8
        3. Poly:2
        4. Trees: 8
2. Fat Supply Quantity
    1. Linear Regression Model
        1. K: 10
        2. Poly:5
    2. Ridge Model
        1. C:default value
        2. max_iteration:default value
        3. K:10
        4. Poly: 2
    3. Random Forrest Regression Model
        1. criterion: "MAE"
        2. K: 20
        3. Poly: 3
        4. Trees: 20
3. Protein Supply Quantity
    1. Linear Regression Model
        1. K: 10
        2. Poly: 2
    2. Ridge Model
        1. C:default value
        2. max_iteration: default value
        3. K: 8
        4. Poly: 2
    3. Random Forrest Regression Model
        1. criterion: "MSE"
        2. K: 20
        3. Poly: 3
        4. Trees: 8
## 5. Results
 mean of cross validation score  
 fat quantity: 0.34  
 food kcal:0.48  
 protein quantity: 0.27  
 dummy regressor: -0.067