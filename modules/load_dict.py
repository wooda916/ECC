import numpy as np

def load_dict(filepath):
    data = np.load(filepath, allow_pickle=True).item()
    return data