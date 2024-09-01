# Fraud Model Development

This repo contains code to develop an XGBoost fraud detection model. First, raw mock data is generated. Next, GCP's python API is used to transform the data and generate features in BigQuery with a sequence of parallelized queries. Finally, the model is trained in python using the XGBoost package. See ./training/results/1.0 for an extensive analysis including:
- Model evaluation (PR curve, ROC curve, score distribution, KS-statistic, n_estimators vs metrics, etc.)
- Model explainability (shapely, PSI/CSI, VIF, WoE/IV, tree-based/permutation importance, correlation)
- Model calibration (Isotonic regression to make the score output approximate a probability of fraud)
- Bayesian hyperparameter tuning (using the Adaptive Tree Parzen Estimator algorithm) 
- Model objects
- Model output scores
- Logs

This model is then implemented (see https://github.com/sthobbs/fraud-model-serving) as a low-latency streaming job on GCP Dataflow using the Apache Beam Java SDK. Once the streaming job is deployed, this repo also has code to pass raw data to the model serving job (via Pub/Sub) and listen for the results. The features and scores from the Dataflow job are then compared to their BigQuery/python-generated counterparts to ensure the model was implemented correctly. Note that java convention of camelCase is often used in python and SQL to match the java pipeline field names in the associated fraud-model-serving repo.

#### Code Structure

./data 
- Raw data and code to generate it
- Feature data
- Scored transaction data

./gcp_helpers
- Helper classes for interacting with GCP's Python API, specifically for:
    - BigQuery
    - Pub/Sub
    - Storage
    - Logging

./gen_features
- Queries/code to generate monthly customer profile tables
- Queries/code to generate model features

./training
- Code for running reproducible machine learning experiments, including:
    - Hyperparameter tuning (using grid search, random search, tpe, or atpe) (with optional cross validation)
    - Model evaluation (PR curve, ROC curve, score distribution, KS-statistic, various tables, etc.)
    - Model explainability (shapely, PSI/CSI, VIF, WoE/IV, permutation feature importance, correlation matrix)
    - Model calibration
    - XGBoost-related objects (plots of n_estimators vs performance metrics, tree-based feature importance)
- See ./training/results/1.0 for detailed experiment results

./utils
-  Multi-threading helper functions.

./validate_serving
- Code to pass data through the fraud-model-serving dataflow pipeline via Pub/Sub
- Code to compare dataflow-generated features and scores, to those generated using BigQuery and python from the same raw data.

### Prerequisits
#### Prerequisits for running develop-model.py
- A GCP project and storage bucket (added to ./config.py)
- A GCP service account, with account key in ./service_account_key.json
- An environment with the packages in requirements.txt

#### Prerequisits for running validate-serving.py
- Create Pub/Sub topics/subscriptions for input and output data (added to ./config.py)
- Deploy Dataflow pipeline from https://github.com/sthobbs/fraud-model-serving

### Running the code
First, run develop-model.py to:
1. Generate mock data of legitimate and fraudulent activity
2. Generate features using BigQuery
3. Train and evaluate a model

Next, with the model serving job deployed, run validate-serving.py to:
1. Pass raw event data to the Pub/Sub input topic
2. Receive scored transaction data from the Pub/Sub output subscription
3. Upload scores to GCS and BigQuery, and process data
4. Run a comparison analysis of features and scores generated in BigQuery/Python vs Dataflow
