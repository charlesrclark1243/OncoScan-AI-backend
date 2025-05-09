# Author: Charles R. Clark
# CS 6440 Spring 2024

from typing import List, Tuple
import pandas as pd
import os

MAPPING = {
    'normal': 0,
    'glioma': 1,
    'meningioma': 2
}

def import_data(dir_path: str, label: int) -> pd.DataFrame:
    img_names = os.listdir(dir_path)

    img_paths = [os.path.join(dir_path, img_name) for img_name in img_names]
    img_labels = [label] * len(img_paths)

    data_dict = {
        'img_path': img_paths,
        'img_label': img_labels
    }

    return pd.DataFrame().from_dict(data=data_dict)

def combine_and_shuffle_data(data_list: List[pd.DataFrame], random_state=42) -> pd.DataFrame:
    combined_data = pd.concat(data_list)
    shuffled_data = combined_data.sample(frac=1, random_state=random_state, ignore_index=True)

    return shuffled_data

def split_data(df: pd.DataFrame, pct=0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = int(len(df) * pct)
    
    train_df = df.iloc[:n, :]
    test_df = df.iloc[n:, :]

    return train_df, test_df