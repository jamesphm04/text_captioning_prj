from pathlib import Path

def get_config():
    return {
        "batch_size": 10,
        "num_epochs": 10,
        "lr": 0.0001,
        "max_length": 40,
        "d_model": 64,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.1,
        "datasource": 'flickr8k',
        "model_folder": "weights",
        "model_basename": "cap_model_",
        "preload": "latest",
        "tokenizer_file": "tokenizer.json"
    }
    
def get_weights_file_path(config, version):
    model_folder = f'{config["datasource"]}_{config["model_folder"]}'
    model_filename = f'{config["model_basename"]}{version}.h5'
    return str(Path('.')/model_folder/model_filename)

def latest_weights_file_path(config):
    model_folder = f'{config["datasource"]}_{config["model_folder"]}'
    model_filename = f'{config["model_basename"]}*.h5'
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    
    weights_files.sort()
    return str(weights_files[-1])