import json
from typing import List, Dict, Optional


class ConfigManager:
    def __init__(self):
            self.config_file = "config.json"
            self.config = self.read_config()
    
    def read_config(self):
        with open(self.config_file, 'r') as file:
            return json.load(file)
    
    def get_config(self, key):
        return self.config.get(key)
    
   
# Example usage
config_manager = ConfigManager()
print(config_manager.get_config('dataset_location'))