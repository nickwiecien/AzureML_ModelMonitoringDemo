import logging
import os
import json
import mlflow
from io import StringIO
import pandas as pd
import json
from azureml.ai.monitoring import Collector

def init():
    global model, inputs_collector, outputs_collector, inputs_outputs_collector

    # "model" is the path of the mlflow artifacts when the model was registered. For automl
    # models, this is generally "mlflow-model".
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = mlflow.sklearn.load_model(model_path)
    inputs_collector = Collector(name='model_inputs')                    
    outputs_collector = Collector(name='model_outputs')
    inputs_outputs_collector = Collector(name='model_inputs_outputs') #note: this is used to enable Feature Attribution Drift
    
    print(inputs_collector)

def run(raw_data):
    json_data = json.loads(raw_data)
    
    scoring_data = json_data['data']
    
    input_df = pd.DataFrame(scoring_data)
    
    print(input_df)
    
    context = inputs_collector.collect(input_df)
    
    predictions = model.predict(input_df)
    
    output_df = input_df
    
    output_df['Predicted_Temperature'] = predictions
    
    outputs_collector.collect(output_df[['Predicted_Temperature']], context)

    inputs_outputs_collector.collect(output_df, context)
    
    result = list(predictions)
    
    print(result)
    
    return result