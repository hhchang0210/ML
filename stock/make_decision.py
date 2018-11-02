#!/usr/bin/env python
import json
from datetime import datetime, timedelta
import sys
import csv
import glob
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics
from shutil import copyfile




src= sys.argv[1]
dst   = sys.argv[2]
copyfile(src + "/result.json", dst)