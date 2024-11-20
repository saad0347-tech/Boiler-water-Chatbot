# -*- coding: utf-8 -*-
"""BoilerWater.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1c_vrTm1TftnLQqrz2jyE2Zc53ZK8sPh4
"""

import pandas as pd

# Upload CSV file
from google.colab import files
uploaded = files.upload()  # Remove the target path argument

# Load data from uploaded CSV file
# Get the uploaded filename from the 'uploaded' dictionary
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# Print first few rows of the dataframe
print(df.head())

# Rest of the data processing code remains the same

import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import streamlit as st


import streamlit as st
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
!pip install streamlit

import subprocess
import sys

required_packages = ['streamlit', 'pandas', 'transformers', 'torch']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

!pip install pandas transformers torch

import streamlit as st

!pip show streamlit

!pip install --upgrade streamlit

!cat requirements.txt

!pip install -r requirements.txt
