# save model params 


def get_params():

    model_params = {
        "RandomForest" : {"n_estimators":100 , "max_depth":5} , 
        "SVC" : {"C":1 , "kernel":"rbf"} , 
        "KNN" : {"n_neighbors": 5} , 
        "DecisionTree" : {"max_depth":5}
    }


    return model_params