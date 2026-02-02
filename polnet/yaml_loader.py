import os
import yaml

def yaml_loader(yaml_path: str, key: str = "polnet") -> dict:
    """
    Load a YAML configuration file into a Python dictionary.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        section (str): Top-level key to extract from the YAML. 

    Returns:
        dict: Nested dictionary representing the YAML configuration.
    """

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file {yaml_path} not found.")
    
    with open(yaml_path, 'r') as f:

        config = yaml.safe_load(f)

    print(config)
    if key not in config: 
        raise ValueError(f"Key {key} is not found in YAML file {yaml_path}.")

    return config[key]
