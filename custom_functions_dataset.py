import numpy as np
#import pandas as pd

# fashion_data = pd.read_csv('fashion_set1.csv')

def clean_and_convert(embedding_str):
    cleaned_str = embedding_str.strip("['").strip("']").replace("\n", "").replace("  ", " ")
    str_list = cleaned_str.split()
    float_array = [float(item) for item in str_list]
    float_np_array = np.array(float_array)
    return float_np_array

# fashion_data['text_features'] = fashion_data['text_features'].apply(clean_and_convert)
# fashion_data['image_features'] = fashion_data['image_features'].apply(clean_and_convert)

