import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def Play_predictor(data_path):
    #Load the data
    data=pd.read_csv(data_path,index_col=0)

    print("Size of actual dataset:",len(data))
    print("")

    #Clean,Prepare & Manipulate data
    Features_name=['Wether','Temperature']

    print("Name of the features:",Features_name)
    print("")

    wether=data.Whether
    Temperature=data.Temperature
    play=data.Play

    #Creating Label Encoder
    le=preprocessing.LabelEncoder()

    #Converting string label into numbers
    wether_encoded=le.fit_transform(wether)
    print(wether_encoded)
    print("")

    Temp_encoded=le.fit_transform(Temperature)
    label=le.fit_transform(play)

    print(Temp_encoded)
    print("")

    #Combining wether and Temperature into single list of tuple
    Features=list(zip(wether_encoded,Temp_encoded))

    #Train data
    model=KNeighborsClassifier(n_neighbors=3)

    #Train data model using the training data sets
    model.fit(Features,label)

    #Test data
    predicted=model.predict([[0,2]])
    print(predicted)
    print("")

    if predicted==1:
        print("You are allowed for playing")

    else:
        print("You are not allowed for playing")
        
def main():
    print("Machine Learning Application")
    print("Play Predictor application using K Nearest Neighbour algorithm")

    Play_predictor('PlayPredictor.csv')

if __name__=="__main__":
    main()