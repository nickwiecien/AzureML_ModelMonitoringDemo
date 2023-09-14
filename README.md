# Azure Machine Learning - Model Monitoring Demo

Welcome to this repository. It is designed as a practical demonstration of [Azure Machine Learning's Model Monitoring](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2) and [Model Data Collection capabilities](https://learn.microsoft.com/en-us/azure/machine-learning/concept-data-collection?view=azureml-api-2) using Jupyter notebooks. While the demonstration focuses on a specific use case, the principles and techniques presented can be adapted and extended for a wide variety of model deployment and monitoring purposes.

## Repository Objective

This repository provides a step-by-step guide to using Azure Machine Learning for model monitoring. It includes a series of Jupyter notebooks that will walk you through:

1. Uploading and registering data in the Azure Machine Learning workspace.
2. Training a custom model using the registered datasets with Mlflow and Scikit-Learn.
3. Deploying the trained model to a Managed Online Endpoint within Azure ML, a feature designed to dynamically scale to accommodate high-traffic workloads.
4. Configuring model data collection to capture all incoming data and model predictions for detailed analysis.
5. Setting up an Azure Machine Learning model/data monitor to evaluate the incoming data and identify any drift in the source features or predictions.
6. Simulating real-world production data by submitting data to the endpoint in a staggered manner.
  
## Data Overview

The demonstration uses a sample of weather data collected between January and October 2019. This data is used to train a model that predicts temperature based on a variety of environmental factors. The dataset was specifically chosen for its features that naturally drift over time, providing a practical demonstration of Azure Machine Learning's drift detection capabilities.

## Utilization Guidelines

To get the most out of this repository, run the notebooks in the provided sequence. Each notebook is numbered and named in line with its specific role in the Azure Machine Learning process. Be sure to replace placeholders with your actual data where necessary.
