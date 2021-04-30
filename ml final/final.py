import json_lines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#read into python
x=[]
y=[]
z=[]
with open('reviews_80.jl','rb') as f:
    for item in json_lines.reader(f):
        x.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])


def tokenize(text):
    toks = word_tokenize(text)
    s_toks = []
    for t in toks:
        s_toks.append(PorterStemmer().stem(t))
    return s_toks
sel_df_max = 0.1
vectorizer = TfidfVectorizer(
    stop_words=nltk.corpus.stopwords.words('english'),
    max_df=sel_df_max,
    tokenizer=tokenize)
X = vectorizer.fit_transform(x)

#clean data
"""
import re
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

x_clean = preprocess_reviews(x)
#vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(norm=None,max_df =0.1)
X = vectorizer.fit_transform(x_clean)
"""


#split the data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)
Xtrain, Xtest, ztrain, ztest = train_test_split(X, z, test_size=0.1)
#logistic regression
    #select best hyperparameter
        #for y
"""
mean_error_y=[]
std_error_y=[]
crange_y=[0.01, 0.05, 0.25, 0.5, 1, 5, 10, 15, 20]
for c in crange_y:
    lr = LogisticRegression(C=c)
    lr.fit(Xtrain, ytrain)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(lr, X, y, cv=5, scoring='f1')
    mean_error_y.append(np.array(scores).mean())
    std_error_y.append(np.array(scores).std())
    print("Final Accuracy: %s"% accuracy_score(ytest, lr.predict(Xtest)))
import matplotlib.pyplot as plt
plt.rc('font', size = 18)
plt.errorbar(crange_y, mean_error_y, yerr=std_error_y, linewidth=3)
plt.xlabel('c'); plt.ylabel('F1 score')
plt.show()
"""

        #final lr
lr_y = LogisticRegression(C=1)
lr_y.fit(Xtrain, ytrain)
fig1=plt.figure()
dispy1=plot_confusion_matrix(lr_y, Xtest, ytest)
print("Logistic Regression confusion matrix for voted up:")
print(dispy1.confusion_matrix)
plt.title("Logistic Regression confusion matrix for voted up")
lr_z = LogisticRegression(C=15)
lr_z.fit(Xtrain, ztrain)
dispz1=plot_confusion_matrix(lr_z, Xtest, ztest)
print("Logistic Regression confusion matrix for early access:")
print(dispz1.confusion_matrix)
plt.title("Logistic Regression confusion matrix for early access")
"""
mean_error_y=[]
std_error_y=[]
crange_y=[0.01, 0.05, 0.25, 0.5, 1, 5, 10, 15, 20]
for c in crange_y:
    lr = LogisticRegression(C=c)
    lr.fit(Xtrain, ztrain)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(lr, X, z, cv=5, scoring='f1')
    mean_error_y.append(np.array(scores).mean())
    std_error_y.append(np.array(scores).std())
    import matplotlib.pyplot as plt

plt.rc('font', size = 18)
plt.errorbar(crange_y, mean_error_y, yerr=std_error_y, linewidth=3)
plt.xlabel('c'); plt.ylabel('F1 score')
plt.show() """ """
#SVM
    #for y select best C value in SVC
   
mean_error=[]
std_error=[]
Ci_range = [0.001,0.1,1,3,5,8,10]
for Ci in Ci_range:
    model = SVC(C=Ci, kernel='rbf')
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
import matplotlib.pyplot as plt
plt.rc('font', size=18);
plt.errorbar(Ci_range,mean_error,yerr=std_error,linewidth=3)
plt.xlabel('Ci'); plt.ylabel('F1 Score')
plt.show()
print("finish")"""
        #the final model
svm_y=SVC(C=5, kernel='rbf',probability=True).fit(Xtrain,ytrain)
dispy2=plot_confusion_matrix(svm_y, Xtest, ytest)
print("SVM confusion matrix for voted up:")
print(dispy2.confusion_matrix)
plt.title("SVM confusion matrix for voted up")
svm_z=SVC(C=1000,kernel='rbf',probability=True).fit(Xtrain,ztrain)
dispz2=plot_confusion_matrix(svm_y, Xtest, ztest)
print("SVM confusion matrix for early access:")
print(dispz2.confusion_matrix)
plt.title("SVM confusion matrix for early access:")
"""
    # for z select best C value in SVC
mean_error=[]
std_error=[]
Ci_range = [0.001,0.1,1,3,5,8,10,100,500,1000,2000]
for Ci in Ci_range:
    model = SVC(C=Ci, kernel='rbf')
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, z, cv=5, scoring='f1')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
import matplotlib.pyplot as plt
plt.rc('font', size=18);
plt.errorbar(Ci_range,mean_error,yerr=std_error,linewidth=3)
plt.xlabel('Ci'); plt.ylabel('F1 Score')
plt.show()
"""
#KNN
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
#final KNN
knn_y=KNeighborsClassifier(n_neighbors=200).fit(Xtrain, ytrain)
dispy3=plot_confusion_matrix(knn_y, Xtest, ytest)
print("KNN confusion matrix for voted up:")
print(dispy3.confusion_matrix)
plt.title("KNN confusion matrix for voted up")

knn_z=KNeighborsClassifier(n_neighbors=1).fit(Xtrain, ztrain)
dispz3=plot_confusion_matrix(svm_y, Xtest, ztest)
print("KNN confusion matrix for early access:")
print(dispz3.confusion_matrix)
plt.title("KNN confusion matrix for early access:")
"""
mean_error=[]
std_error=[]
n_range = [1,2,3,4,5,6,7,8,9,10,12,15,20,40,50]
from sklearn import metrics

for n in n_range:
    model_knn = KNeighborsClassifier(n_neighbors=n).fit(Xtrain, ytrain)
    y_pred=model_knn.predict(Xtest)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model_knn, X, y, cv=5, scoring='f1')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
    print(metrics.accuracy_score(ytest, y_pred))
import matplotlib.pyplot as plt
plt.rc('font', size=18);
plt.errorbar(n_range,mean_error,yerr=std_error,linewidth=3)
plt.xlabel('n'); plt.ylabel('F1 Score')
plt.show()
"""
#baseline model:
from sklearn.dummy import DummyClassifier
dummy_y = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
dispy4=plot_confusion_matrix(svm_y, Xtest, ytest)
print("Dummy classifier matrix for voted up:")
print(dispy4.confusion_matrix)
plt.title("dummyclassifier confusion matrix for voted up")
dummy_z = DummyClassifier(strategy="most_frequent").fit(Xtrain, ztrain)
dispz4=plot_confusion_matrix(svm_y, Xtest, ztest)
print("dummyclassifier matrix for early access:")
print(dispz4.confusion_matrix)
plt.title("dummyclassifier confusion matrix for early access:")
plt.show()

#ROC curve
#for y
from sklearn.metrics import roc_curve,auc

fpr1, tp1, _ =roc_curve(ytest,lr_y.decision_function(Xtest))
plt.plot(fpr1,tp1,label='logistic regression',c='b',linewidth=6)

y_scores_knn = knn_y.predict_proba(Xtest)
fpr2, tp2, threshold1 = roc_curve(ytest, y_scores_knn[:, 1])
plt.plot(fpr2,tp2,label='KNN',c='y')

y_scores_svm = svm_y.predict_proba(Xtest)
fpr3, tp3, threshold2 = roc_curve(ytest, y_scores_svm[:, 1])
plt.plot(fpr3,tp3,label='svm',c='black')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot([0, 1], [0, 1], color='g',linestyle='--')

#baseline
y_scores_dummy = dummy_y.predict_proba(Xtest)
fpr4, tp4, threshold3 = roc_curve(ytest, y_scores_dummy[:, 1])
plt.plot(fpr4,tp4,label='baseline model:dummy classifier',c='r',linewidth=6)
plt.title("ROC curve for voted up")
plt.legend()
plt.show()