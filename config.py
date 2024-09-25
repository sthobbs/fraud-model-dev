###################################
##### Mock Data Configuration #####
###################################

start_date = '2021-01-01'         # start date of sessions
end_date = '2023-01-01'           # end date of sessions
n_distinct_actions = 10           # number of distinct general actions (e.g. 'action_0')
n_customers = 500                 # number of distinct customers
n_fraudsters = 100                # number of distinct fraudsters
n_sessions = 5000                 # number of sessions
n_legit_recipients = 1000         # number of distinct legit recipients
n_fraud_recipients = 100          # number of distinct fraud recipients
fraud_session_rate = 0.1          # approx proportion of sessions that are fraud
seed = 123                        # random seed
raw_data_dir = './data/raw/data'  # path to directory where raw mock data will be saved
save_formats = ['json']           # formats to save data in ('csv', 'json', 'json_full')


#####################################
##### Feature Gen Configuration #####
#####################################

project_id = 'analog-arbor-367702'  # Google Cloud project ID
dataset_id = 'fraud_detection'      # BigQuery dataset ID
bucket_name = 'test-bucket-85203'   # GCS bucket name
profile_dir = './data/profiles'     # path to directory where profile data will be saved

query_params = {
    'project_id': project_id,
    'dataset_id': dataset_id,
    'start_date': start_date,
    'end_date': end_date, 
    
    # out of time train/validaiton/test split
    'train_start_date': '2022-01-01',  
    'train_end_date': '2022-11-01',
    'valid_start_date': '2022-11-01',  
    'valid_end_date': '2022-12-01',
    'test_start_date': '2022-12-01',  
    'test_end_date': '2023-01-01',
}


##################################################
##### Model Serving Validation Configuration #####
##################################################

test = True         # if True, test data will be used in scoring job
mongo_setup = True  # if True, customer profiles and indexes will be reset in MongoDB

input_topic = 'test-input'               # event streamer will publish raw data to this topic
input_subscription = 'test-input-sub2'   # scoring job will subscribe to this subscription
output_topic = 'test-output'             # scoring job will publish to this topic
output_subscription = 'test-output-sub'  # event streamer will subscribe to this subscription

n_processes = 8  # number of processes to run in parallel
n_threads = 16   # number of threads per process to run in concurrently

scored_data_dir = './data/scores'                         # path to directory where scored output data will be saved
model_exp_dir = './training/results/1.0-20240905-111903'  # path to directory where model artifacts will be saved
model_path = f"{model_exp_dir}/model/model.pkl"           # path to model artifact
model_id = 'txn_model.v2'                                 # model id specified in score events

# event streamer configuration
test_data_input_path = f"{raw_data_dir}/events.json"              # path to test input data
test_data_output_path = f"{scored_data_dir}/serving_scores.json"  # path to scored output test data
event_streamer_delay = 0.001                                      # delay in seconds between events
pause_before_feature_gen = 0                                    # delay in seconds to give MongoDB time to process events

# MongoDB configuration
mongo_client_uri = "mongodb://localhost:27017/"  # MongoDB connection string
db_name = 'fraud'                                # name of database in MongoDB
event_collection = 'events'                      # name of collection in MongoDB for raw events
