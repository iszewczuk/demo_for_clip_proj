import numpy as np

def clean_and_convert(embedding_str: str) -> np.ndarray:
    cleaned_str = embedding_str.strip("['").strip("']").replace("\n", "").replace("  ", " ")
    str_list = cleaned_str.split()
    float_array = [float(item) for item in str_list]
    float_np_array = np.array(float_array)
    return float_np_array
