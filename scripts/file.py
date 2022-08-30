import numpy as np
import pandas as pd

class Myfile:
  
  def __init__(self):
    pass
  
  def read_csv(self, path):
    try:
        df = pd.read_csv(path)
        print("--> file read as csv")
        return df
    except FileNotFoundError:
        print("--> file not found")
  
  
  def save_csv(self, df, path):
    try:
        df.to_csv(path, index=False)
        print('--> File Successfully Saved.!!!')

    except Exception:
        print("--> File Save failed...")

    return df