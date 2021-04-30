import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor


# get details of food in different files and make prediction about the probability of obesity
# fill N/A
# give a constant mean value when there is a “NA” of value
def get_features(file):
    obesity = file['Obesity']
    features = file.iloc[:, 1:24]
    # print(len(obesity.dropna())/len(obesity)) 98%, we do not need to drop those features with N/A
    obesity.fillna(obesity.mean(), inplace=True)
    return features, obesity


def get_training_data():
    file_fat_kal = pandas.read_csv("food_supply_kcal_data/Food_Supply_kcal_Data.csv")
    file_protein_supply = pandas.read_csv("protein_supply_quantity_data/Protein_Supply_Quantity_Data.csv")
    file_fat_quantity = pandas.read_csv("fat_supply_quantity_data/Fat_Supply_Quantity_Data.csv")
    x, y = get_features(file_protein_supply)
    return np.array(x), np.array(y).reshape(-1, 1)


def get_KFold(x, y, model_type: str):
    k_list = [2, 4, 6, 8, 10, 20, 40]
    poly = PolynomialFeatures(degree=3)
    x_ = poly.fit_transform(x)
    score_list = []
    train_error_mse = []
    test_error_mse = []
    train_error_std = []
    test_error_std = []
    for k in k_list:
        kf = KFold(n_splits=k)
        temp_train = []
        temp_test = []
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'random_forrest':
            model = RandomForestRegressor()
        elif model_type == 'ridge':
            model = Ridge()
        for train, test in kf.split(x_):
            model.fit(x_[train], y[train])
            temp_train.append(mean_squared_error(y[train], model.predict(x_[train])))
            temp_test.append(mean_squared_error(y[test], model.predict(x_[test])))
        score_list.append(model.score(x_, y))
        train_error_mse.append(np.array(temp_train).mean())
        test_error_mse.append(np.array(temp_test).mean())
        train_error_std.append(np.array(temp_train).std())
        test_error_std.append(np.array(temp_test).std())
    plot_mse_and_score(score_list, train_error_std, test_error_std, train_error_mse, test_error_mse, k_list,
                       model_type, 'K')


def get_poly(x, y, model_type: str):
    kf = KFold(n_splits=10)
    degree_list = [2, 3, 4, 5, 6]
    score_list = []
    train_error_mse = []
    test_error_mse = []
    train_error_std = []
    test_error_std = []
    for poly_degree in degree_list:
        temp_train = []
        temp_test = []
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'random_forrest':
            model = RandomForestRegressor()
        elif model_type == 'ridge':
            model = Ridge()
        poly = PolynomialFeatures(degree=poly_degree)
        x_ = poly.fit_transform(x)
        for train, test in kf.split(x_):
            model.fit(x_[train], y[train])
            temp_train.append(mean_squared_error(y[train], model.predict(x_[train])))
            temp_test.append(mean_squared_error(y[test], model.predict(x_[test])))
        score_list.append(model.score(x_, y))
        train_error_mse.append(np.array(temp_train).mean())
        test_error_mse.append(np.array(temp_test).mean())
        train_error_std.append(np.array(temp_train).std())
        test_error_std.append(np.array(temp_test).std())
    plot_mse_and_score(score_list, train_error_std, test_error_std, train_error_mse, test_error_mse, degree_list,
                       model_type, 'poly degree')


def plot_mse_and_score(score_list, train_error_std, test_error_std, train_error_mean, test_error_mean, x_list,
                       model_type, x_list_name):
    if model_type == 'random_forrest' and x_list_name == 'criterion':
        plt.xticks([0, 1], x_list)
    axs1 = plt.subplot(211)
    plt.errorbar(x_list, train_error_mean, yerr=train_error_std, label='train_mse')
    plt.errorbar(x_list, test_error_mean, yerr=test_error_std, label='test_mse')
    plt.title(model_type + ' Error VS ' + x_list_name)
    plt.xlabel(x_list_name)
    plt.ylabel('error')
    plt.legend()
    axs2 = plt.subplot(212)
    plt.plot(x_list, score_list, marker='o')
    plt.xlabel(x_list_name)
    plt.ylabel('score')
    plt.title('score in different ' + x_list_name)
    plt.show()


def get_lr_prediction(x, y):
    lr = LinearRegression()
    poly = PolynomialFeatures(degree=3)
    x, y = get_training_data()
    lr.fit(poly.fit_transform(x), y)


# best k=8, best poly=4
def get_ridge_C(x, y):
    kf = KFold(n_splits=8)
    poly = PolynomialFeatures(degree=4)
    x_ = poly.fit_transform(x)
    C_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    score_list = []
    train_error_mse = []
    test_error_mse = []
    train_error_std = []
    test_error_std = []
    for c in C_list:
        temp_train = []
        temp_test = []
        model = Ridge(alpha=1 / (2 * c))
        for train, test in kf.split(x_):
            model.fit(x_[train], y[train])
            temp_train.append(mean_squared_error(y[train], model.predict(x_[train])))
            temp_test.append(mean_squared_error(y[test], model.predict(x_[test])))
        score_list.append(model.score(x_, y))
        train_error_mse.append(np.array(temp_train).mean())
        test_error_mse.append(np.array(temp_test).mean())
        train_error_std.append(np.array(temp_train).std())
        test_error_std.append(np.array(temp_test).std())
    plot_mse_and_score(score_list, train_error_std, test_error_std, train_error_mse, test_error_mse, C_list,
                       "ridge", 'C')


# best k=8, best poly=4
def get_ridge_max_iteration(x, y):
    kf = KFold(n_splits=8)
    poly = PolynomialFeatures(degree=4)
    kf = KFold(n_splits=8)
    poly = PolynomialFeatures(degree=4)
    x_ = poly.fit_transform(x)
    max_iteration_list = [1e3, 1e4, 1e5, 1e6]
    score_list = []
    train_error_mse = []
    test_error_mse = []
    train_error_std = []
    test_error_std = []
    for iteration in max_iteration_list:
        temp_train = []
        temp_test = []
        model = Ridge(max_iter=iteration)
        for train, test in kf.split(x_):
            model.fit(x_[train], y[train])
            temp_train.append(mean_squared_error(y[train], model.predict(x_[train])))
            temp_test.append(mean_squared_error(y[test], model.predict(x_[test])))
        score_list.append(model.score(x_, y))
        train_error_mse.append(np.array(temp_train).mean())
        test_error_mse.append(np.array(temp_test).mean())
        train_error_std.append(np.array(temp_train).std())
        test_error_std.append(np.array(temp_test).std())
    plot_mse_and_score(score_list, train_error_std, test_error_std, train_error_mse, test_error_mse, max_iteration_list,
                       "ridge", 'iteration')


def get_ridge_prediction(x, y):
    pass


# best k=8,poly=2
def get_number_of_trees(x, y):
    kf = KFold(n_splits=8)
    poly = PolynomialFeatures(degree=2)
    x_ = poly.fit_transform(x)
    trees_list = [2, 4, 6, 8, 10, 20, 40, 60, 100]
    score_list = []
    train_error_mse = []
    test_error_mse = []
    train_error_std = []
    test_error_std = []
    for tree in trees_list:
        temp_train = []
        temp_test = []
        model = RandomForestRegressor(n_estimators=tree)
        for train, test in kf.split(x_):
            model.fit(x_[train], y[train])
            temp_train.append(mean_squared_error(y[train], model.predict(x_[train])))
            temp_test.append(mean_squared_error(y[test], model.predict(x_[test])))
        score_list.append(model.score(x_, y))
        train_error_mse.append(np.array(temp_train).mean())
        test_error_mse.append(np.array(temp_test).mean())
        train_error_std.append(np.array(temp_train).std())
        test_error_std.append(np.array(temp_test).std())
    plot_mse_and_score(score_list, train_error_std, test_error_std, train_error_mse, test_error_mse, trees_list,
                       "random_forrest", 'trees')


# best k=8,poly=2
def get_criterion(x, y):
    kf = KFold(n_splits=8)
    poly = PolynomialFeatures(degree=2)
    x_ = poly.fit_transform(x)
    criterion_list = ['mae', 'mse']
    score_list = []
    train_error_mse = []
    test_error_mse = []
    train_error_std = []
    test_error_std = []
    for critic in criterion_list:
        temp_train = []
        temp_test = []
        model = RandomForestRegressor(criterion=critic)
        for train, test in kf.split(x_):
            model.fit(x_[train], y[train])
            temp_train.append(mean_squared_error(y[train], model.predict(x_[train])))
            temp_test.append(mean_squared_error(y[test], model.predict(x_[test])))
        score_list.append(model.score(x_, y))
        train_error_mse.append(np.array(temp_train).mean())
        test_error_mse.append(np.array(temp_test).mean())
        train_error_std.append(np.array(temp_train).std())
        test_error_std.append(np.array(temp_test).std())
    plot_mse_and_score(score_list, train_error_std, test_error_std, train_error_mse, test_error_mse, criterion_list,
                       "random_forrest", 'criterion')


if __name__ == '__main__':
    model_and_parameters = {"linear_regression": ['polynomial features\' degree', 'Number of KFold'],
                            "ridge": ['max iteration', 'C', 'polynomial features\' degree', 'Number of KFold'],
                            "random_forrest": ['number of trees', 'criterion', 'polynomial features\' degree',
                                               'Number of KFold']}
    print('This is the Group70 week5 assignment')
    x, y = get_training_data()
    # get_poly(x, y, list(model_and_parameters.keys())[0])
    # get_KFold(x, y, list(model_and_parameters.keys())[0])
    # get_poly(x, y, list(model_and_parameters.keys())[1])
    # get_KFold(x, y, list(model_and_parameters.keys())[1])
    # get_poly(x, y, list(model_and_parameters.keys())[2])
    # get_KFold(x, y, list(model_and_parameters.keys())[2])
    # get_ridge_C(x, y)
    # get_ridge_max_iteration(x,y)
    # get_criterion(x,y)
    # get_number_of_trees(x, y)
    dummy = DummyRegressor()
    poly = PolynomialFeatures(degree=3)

    random_food_supply = RandomForestRegressor(criterion='mse', n_estimators=8)
    from sklearn.model_selection import cross_val_score

    x = poly.fit_transform(x)
    random_food_supply.fit(x, y)
    dummy_score = cross_val_score(dummy, x, y, cv=10)
    fat_quantity_score = cross_val_score(random_food_supply, x, y, cv=20)
    food_supply_score = cross_val_score(random_food_supply, x, y)

    dummy_score = cross_val_score(dummy,x,y,cv=10)
    print(np.array(dummy_score).mean())
