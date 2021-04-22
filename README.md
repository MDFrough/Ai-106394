# Ai-106394 : Project #
### PROJECT MEMBERS ###
StdID | Name
------------ | -------------
**63866** | **Mohammad Dursham** 
63924 | Misbah Shakeel
62606 | Muhammad Azhar
## Assignemnt Description ##
Apply 6 different convolution on the features to get 6 different data sets.
The 6 convolutions are based on the three different sizes.

    a. Convolution 5x5
    b. Convolution 7x7
    c. Convolution 9x9

and 2 filters

            1, 1, 1, 1, 1
            1, 1, 1, 1, 1
            1, 1, 1, 1, 1
            1, 1, 1, 1, 1
            1, 1, 1, 1, 1

            ----

            1, 1, 1, 1, 1
            1, 2, 2, 2, 1
            1, 2, 3, 2, 1
            1, 2, 2, 2, 1
            1, 1, 1, 1, 1

## Screenshot of highest score on kaggle ##
![Screenshot_1](https://user-images.githubusercontent.com/61589320/115714389-819b7880-a390-11eb-8965-c19dc0dcbc65.png)
![Screenshot_2](https://user-images.githubusercontent.com/61589320/115714413-86f8c300-a390-11eb-9aad-9d27a4110870.png)

## Models Used ##
    a. MultinomialNB
    b. Linear Regression Classification
    c. SVM
    d. KNN

### KNN ###
The k-nearest neighbors (KNN) algorithm can be used to solve both classification and regression problems. in knn we have used 3 as nearest neighbor to better satisfy our need.

### SVM ###
Support Vector Machine (SVM) algorithm can be used for both classification or regression challenges.

### Linear Regression ###
Linear Regression algorithm is used to performs a regression task.

### MultinomialNB ###
MultinomialNB algorithm is used in Text Classification, Spam filtering and Sentiment Analysis.

## References ##
- Documentation of python : https://docs.python.org/3/
- Documentation of scikit-learn : https://scikit-learn.org/stable/user_guide.html
- Explanation of CategoricalNB : https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html
- Explanation of SVM : https://scikit-learn.org/stable/modules/svm.html
- Explanation of KNN : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- Explanation of LinearRegression : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
