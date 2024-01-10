def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 100,
        "lr": 0.001,
        "seq_len": 20,
        "model_folder": "weights",
        "model_basename": "cap_model_",
        "preload": "latest",
        "tokenizer_file": "tokenizer.json"
    }