""" main file to run """

import mlflow
from src.data_preperation import load_and_split_data
from src.model_training import train_model
from src.models import get_models
from src.model_evaluate import evaluate_model
from src.model_params import get_params

def main():
    
    # set exper
    mlflow.set_experiment("Iris_Classification2")

    # load & split data 
    x_train , x_test , y_train , y_test = load_and_split_data()

    # models
    models = get_models()

    # params
    params = get_params()

    # train & evaluate each model
    for model_name , model in models.items():
        with mlflow.start_run(run_name=model_name):
            # train 
            tained_model = train_model(x_train,y_train,model,model_name,params[model_name])

            # evaluate
            evaluate_model(tained_model,x_test,y_test)
            


if __name__ == "__main__":
    main()