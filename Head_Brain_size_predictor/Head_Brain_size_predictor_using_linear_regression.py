import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def Head_Brain_Predictor():
    #Load data
    data=pd.read_csv('MarvellousHeadBrain.csv')

    print("Size of data set:",data.shape)

    x=data['Head Size(cm^3)'].values
    y=data['Brain Weight(grams)'].values

    X=x.reshape((-1,1))

    n=len(X)

    reg=LinearRegression()

    reg=reg.fit(X,y)

    y_pred=reg.predict(X)

    r2=reg.score(X,y)

    print(r2)


def main():
    print("Supervised Machine Learning")
    print("Linear Regression on Head Brain size data set")

    Head_Brain_Predictor()

if __name__=="__main__":
    main()