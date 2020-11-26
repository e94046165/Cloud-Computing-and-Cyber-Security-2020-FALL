# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "opt/bitnami/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path

import numpy as np
from pyspark import SparkConf, SparkContext
import pyspark

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "3g")) # Set 3 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    feats.insert(0,label)
    features = [ float(feature) for feature in feats ] # need floats
    return np.array(features)

def sigmoid(s):
    return 1/(1+np.exp(-s))

def predict(w, b, point):
    y = int(point.item(0))
    x = point.take(range(1, point.size))
    yhat = np.dot(w, x) + b
    return sigmoid(yhat)

def gradient(w, b, point):
    y = int(point.item(0))
    x = point.take(range(1, point.size))
    err = y - predict(w, b, point)
    b = err
    w = err * x
    return w, b

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("data_banknote_authentication.txt")
parsedData = data.map(mapper)

def logisticSGD(T, w, b, lr, data):
    for i in range(T):
        data_random = data.sample(True, 0.2, i)
        count = data_random.count()
        g_w = data_random.map(lambda x: gradient(w, b, x)[0]).reduce(lambda x, y: x + y)
        g_b = data_random.map(lambda x: gradient(w, b, x)[1]).reduce(lambda x, y: x + y)
        w = w + lr * g_w / count
        b = b + lr * g_b / count       
        labelsAndPreds = parsedData.map(lambda point: (int(point.item(0)), round(predict(w, b, point))))
        error = labelsAndPreds.filter(lambda res: res[0] != res[1]).count() / float(parsedData.count())
        if i%100 == 0:
            print("Epoch:", i, "Learning Rate:", lr, "Training Error = " + str(error))

data = parsedData
T = 1000
w = np.zeros(data.first().size-1)
b = 0 
lr = 0.01
logisticSGD(T, w, b, lr, data)
