from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    #Load dataset
    wine=datasets.load_wine()

    #Print names of the features
    print(wine.feature_names)

    #Print the label species(class_0,class_1,class_2)
    print(wine.target_names)

    #Print the wine data(top 5 records)
    print(wine.data[:5])

    #Print wine labels(0:class_0,1:class_1,2:class_2)
    print(wine.target)

    #Split dataset into training set and test set
    X_train,X_test,Y_train,Y_test=train_test_split(wine.data,wine.target,test_size=0.3) #70% training and 30% testing

    #Create KNN classifier
    Knn=KNeighborsClassifier(n_neighbors=3)

    #Train the model using using training sets
    Knn.fit(X_train,Y_train)

    #Predict the response for test dataset
    y_pred=Knn.predict(X_test)

    #Model accuracy ,how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(Y_test,y_pred))


def main():
    print("Machine Learning Application")

    print("Wine Predictor application using K Nearest neighbor algorithm")

    WinePredictor()

if __name__=="__main__":
    main()