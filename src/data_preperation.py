# step 1 : 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_and_split_data():
    
    iris = load_iris()
    x,y = iris.data , iris.target

    # split data
    x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=42)

    return x_train , x_test , y_train , y_test