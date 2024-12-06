import json
import os


class ConfigManager:
    def __init__(self, config_file="config.json"):
        """
        Initializes the ConfigManager.

        :param config_file: Path to the configuration file (default: "config.json").
        """
        self.config_file = config_file
        self.config = self.read_config()

    def read_config(self):
        """
        Reads the configuration file and returns the content as a dictionary.

        :return: Dictionary containing configuration.
        :raises FileNotFoundError: If the config file does not exist.
        :raises json.JSONDecodeError: If the file content is not valid JSON.
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")

        try:
            with open(self.config_file, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from '{self.config_file}': {e}")

    def get_config(self, key, default=None):
        """
        Retrieves a value from the configuration.

        :param key: Key to look up in the configuration.
        :param default: Default value to return if the key is not found (default: None).
        :return: The value associated with the key or the default value.
        """
        return self.config.get(key, default)


# Example usage
try:
    config_manager = ConfigManager()
    print(config_manager.get_config('dataset_location', default="config.json"))
except (FileNotFoundError, ValueError) as e:
    print(f"Error: {e}")
