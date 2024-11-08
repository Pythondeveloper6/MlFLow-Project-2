# step 4

import mlflow
from sklearn.metrics import accuracy_score , precision_score , f1_score , recall_score


def evaluate_model(model,x_test,y_test):

    y_pred = model.predict(x_test)

    metrices = {
     "accuracy_score" : accuracy_score(y_test,y_pred)   ,
     'precision_score' : precision_score(y_test,y_pred,average='weighted') , 
     'f1_score': f1_score(y_test,y_pred,average='weighted') , 
     'recall_score': recall_score(y_test,y_pred,average='weighted')
    }

    # save metrices in mlflow
    for metric_name , metric_value in metrices.items():
        mlflow.log_metric(metric_name,metric_value)


    return metrices