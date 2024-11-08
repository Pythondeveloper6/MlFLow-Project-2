# step 3
import mlflow
import mlflow.sklearn

def train_model(x_train,y_train,model,model_name,params):
    
    # set model params
    model.set_params(**params)   # key=value for each param

    # train 
    model.fit(x_train,y_train)

    # save model
    mlflow.sklearn.log_model(model,model_name)

    # log model params
    for param_name , param_value in params.items():
        mlflow.log_param(param_name,param_value)

    return model


