from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def IrisClassifier():
    Dataset=load_iris()

    Data=Dataset.data
    Target=Dataset.target

    Data_train,Data_test,Target_train,Target_test=train_test_split(Data,Target,test_size=0.5)

    classifier=KNeighborsClassifier()

    classifier.fit(Data_train,Target_train)

    predictions=classifier.predict(Data_test)

    Accuracy=accuracy_score(Target_test,predictions)

    return Accuracy

def main():
    Ret=IrisClassifier()

    print("Accuracy of Iris with KNN is:",Ret*100)

if __name__=="__main__":
    main()