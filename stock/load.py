import json
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
import csv
import os
import glob


id_1 = "2330"
id_2 = "3008"
path = '/home/mlb/res/stock/twse/json/'
#print(sys.argv[1])


date_start = sys.argv[1]
date_end  = sys.argv[2]
output     = sys.argv[3] + "/"

datetime_start = datetime.strptime(date_start, '%Y-%m-%d')
datetime_end = datetime.strptime(date_end, '%Y-%m-%d')
diff = (datetime_end - datetime_start).days
#print("fidd=", diff)
schema = ["date", "id", "close", "high", "low", "volume"]

now = datetime.now().strftime("%Y-%m-%d")
#date = datetime.strptime(today, '%Y-%m-%d')
#date = '2018-09-28'

try:  
    os.mkdir(output)
except OSError:  
    print ("Creation of the directory %s failed" % output)
else:  
    print ("Successfully created the directory %s " % output)


filelist = glob.glob(os.path.join(output, "*.csv"))
print("I am hiere")
print(filelist)
for f in filelist:
    print("f=", f)
    os.remove(f)


date = date_end
file_path = output + id_1 + now + '.csv'

with open(file_path, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow([g for g in schema])
    for i in range(diff):
        
        file = path + date
        try: 
   
            with open(file +'.json') as f:
                data = json.load(f)
                #print(date, data[id])
                #print("fhoewhgoe", data, data[id_1])
                writer.writerow([date, id_1, data[id_1]["close"], data[id_1]["high"], data[id_1]["low"], data[id_1]["volume"]])
        except Exception as e:
            print(file, "not exist")   
        
        date = datetime.strptime(date, '%Y-%m-%d')
    
        date = date - timedelta(days=1)
    
        date = date.strftime('%Y-%m-%d')
        

date = date_end
file_path = output + id_2 + now+ '.csv'
with open(file_path, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow([g for g in schema])
    for i in range(diff):
        
        file = path + date
        try: 
   
            with open(file +'.json') as f:
                data = json.load(f)
                #print(date, data[id])
                writer.writerow([date, id_2, data[id_2]["close"], data[id_2]["high"], data[id_2]["low"], data[id_2]["volume"]])
        except Exception as e:
            print(file, "not exist")   
        
        date = datetime.strptime(date, '%Y-%m-%d')
    
        date = date - timedelta(days=1)
    
        date = date.strftime('%Y-%m-%d')
