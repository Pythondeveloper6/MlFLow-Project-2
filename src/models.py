# Step 2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



def get_models():

    return {
        "RandomForest" : RandomForestClassifier() , 
        "SVC" : SVC() , 
        "KNN" : KNeighborsClassifier() , 
        "DecisionTree": DecisionTreeClassifier()
    }