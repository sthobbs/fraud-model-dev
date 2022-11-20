###################################
##### Mock Data Configuration #####
###################################
n_distinct_actions = 10  # number of distinct general actions (e.g. 'action_0')
n_customers = 1000  # number of distinct customers
n_fraudsters = 5  # number of distinct fraudsters
n_sessions = 2000  # number of sessions
n_legit_recipients = 1000  # number of distinct legit recipients
n_fraud_recipients = 100  # number of distinct fraud recipients
fraud_session_rate = 0.1  # approx proportion of sessions that are fraud
start_date = '2021-01-01'  # start date of sessions
end_date = '2023-01-01'  # end date of sessions
raw_data_output_dir = './data/raw/data'  # path to directory where raw mock data will be saved
save_formats = ['json']  # formats to save data in ('csv', 'json', 'json_full')
seed = 123  # random seed

###################################
##### Feature Gen Configuration #####
###################################

project_id = 'analog-arbor-367702'
dataset_id = 'fraud_detection'
bucket_name = 'test-bucket-85203'

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

