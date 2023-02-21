from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def IrisClassifier():
    DataSet=load_iris()

    Data=DataSet.data
    Target=DataSet.target

    Data_train,Data_test,Target_train,Target_test=train_test_split(Data,Target,test_size=0.5)

    classifier=DecisionTreeClassifier()

    classifier.fit(Data_train,Target_train)

    predictions=classifier.predict(Data_test)

    Accuracy=accuracy_score(Target_test,predictions)

    return Accuracy


def main():
    Ret=IrisClassifier()

    print("Accuracy of Iris with Decision Tree Classifier is:",Ret*100)

if __name__=="__main__":
    main()