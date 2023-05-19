import sys
import numpy as np
import pandas as pd
import sklearn as sk
import keras
import tensorflow as tf
import matplotlib as mpl
import scipy as sp
import json
versions = {}

versions['Python'] = sys.version
versions['Numpy'] = np.__version__
versions['Pandas'] = pd.__version__
versions['Matplotlib'] = mpl.__version__
versions['Scikit-learn'] = sk.__version__
versions['Scipy'] = sp.__version__
versions['Tensorflow'] = tf.__version__

with open('versions.json', 'w') as file:
    file.write(json.dumps(versions))

print(f"Python version: {sys.version}")
print('\n')
print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {mpl.__version__}")
print(f"Scikit-learn version: {sk.__version__}")
print(f'Scipy version: {sp.__version__}')
print(f"Keras version: {keras.__version__}")
print(f"Tensorflow version: {tf.__version__}")