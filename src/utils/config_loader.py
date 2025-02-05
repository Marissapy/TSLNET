import json

class ConfigLoader:
    """
    Loads configuration settings from a JSON file.
    """
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)

    def get(self, key, default=None):
        """
        Gets a value from the configuration.
        
        Args:
            key (str): Configuration key.
            default: Default value if the key is not found.
        
        Returns:
            The value associated with the key or the default value.
        """
        return self.config.get(key, default)
