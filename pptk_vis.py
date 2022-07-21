# imports 
import polars as pl
import numpy as np
import os
import pptk
from dotenv import load_dotenv, find_dotenv

# Load in environment file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

#csv_path = os.environ.get('WIND_PATH')
csv_path = os.environ.get('LINUX_PATH')

# python method for extracting the features
def load_csv(csv, x_name, y_name, z_name):

    # Read in csv
    df = pl.read_csv(csv, columns=[x_name, y_name, z_name])
    data = df.to_numpy()

    return data

def foo():
    data = load_csv(csv_path, 'X (nm)', 'Y (nm)', 'Z (nm)')
    #print(data)
    v = pptk.viewer(data)

    return v





