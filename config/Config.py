import json


class Config:
    """
    Generalized configuration class for managing optional features.
    Provides default settings, dynamic updates, validation, and export/import functionality.
    """

    def __init__(self, **kwargs):
        """
        Initialize the configuration with default or custom settings.

        Parameters:
        - kwargs: Key-value pairs for custom configurations.
        """
        # Default configurations
        self.settings = {
            'use_gpu': False,  # Disable GPU acceleration by default
            'parallel': False,  # Disable parallel processing by default
            'dynamic_config': False,  # Disable dynamic configuration by default
            'logging_level': 'INFO',  # Default logging level
        }

        # Update with custom configurations
        self.update(**kwargs)

    def update(self, **kwargs):
        """
        Update the configuration dynamically with validation.

        Parameters:
        - kwargs: Key-value pairs for updated configurations.

        Raises:
        - KeyError: If an invalid configuration key is provided.
        - ValueError: If a configuration value is invalid.
        """
        for key, value in kwargs.items():
            if key in self.settings:
                # Validate specific keys
                if key == 'use_gpu' and not isinstance(value, bool):
                    raise ValueError(f"Invalid value for {key}: Expected a boolean.")
                if key == 'parallel' and not isinstance(value, bool):
                    raise ValueError(f"Invalid value for {key}: Expected a boolean.")
                if key == 'dynamic_config' and not isinstance(value, bool):
                    raise ValueError(f"Invalid value for {key}: Expected a boolean.")
                if key == 'logging_level' and value not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                    raise ValueError(f"Invalid value for {key}: Must be one of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].")

                # Update the setting
                self.settings[key] = value
            else:
                raise KeyError(f"Invalid configuration key: {key}")

    def get(self, key):
        """
        Get the value of a specific configuration key.

        Parameters:
        - key: The configuration key to retrieve.

        Returns:
        - The value of the configuration key.

        Raises:
        - KeyError: If the key does not exist in the configuration.
        """
        if key not in self.settings:
            raise KeyError(f"Configuration key '{key}' does not exist.")
        return self.settings[key]

    def reset(self):
        """
        Reset the configuration to default settings.
        """
        self.settings = {
            'use_gpu': False,
            'parallel': False,
            'dynamic_config': False,
            'logging_level': 'INFO',
        }

    def to_dict(self):
        """
        Export the current configuration as a dictionary.

        Returns:
        - dict: The current configuration settings.
        """
        return self.settings.copy()

    def to_json(self, filepath):
        """
        Export the current configuration to a JSON file.

        Parameters:
        - filepath (str): Path to the JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.settings, f, indent=4)
        print(f"Configuration saved to {filepath}")

    @classmethod
    def from_json(cls, filepath):
        """
        Load configuration from a JSON file.

        Parameters:
        - filepath (str): Path to the JSON file.

        Returns:
        - Config: A new Config object with settings loaded from the file.

        Raises:
        - FileNotFoundError: If the file does not exist.
        - json.JSONDecodeError: If the file is not a valid JSON file.
        """
        with open(filepath, 'r') as f:
            settings = json.load(f)
        return cls(**settings)

    def __repr__(self):
        """String representation of the configuration."""
        return f"Config({self.settings})"