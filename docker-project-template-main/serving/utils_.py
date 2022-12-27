import os
from comet_ml import API

def download_model(username, project_name, experiment_key, model_name,ver):
    api = API(api_key = os.environ.get('COMET_API_KEY'))
    # api.get(username+"/"+project_name+"/"+experiment_key)
    api.download_registry_model(username,model_name,version=ver, output_path="./../models/"+model_name+"/")
    
