import json
from pyspark import SparkContext
import time
import numpy as np
import  xgboost as xgbr
import sys
#from sklearn.metrics import mean_squared_error



def loadBussinessJson(line):
    business = json.loads(line)
    return (business['business_id'],[float(business['stars']),float(business['review_count'])])

def loadUserJson(line):
    user = json.loads(line)
    return (user['user_id'],[float(user['average_stars']),float(user['review_count'])])

def createFeature(input):
    ip_data = input.split(',')
    user = ip_data[0]
    business = ip_data[1]
    if len(ip_data) == 3:
        rating = float(ip_data[2])
    else:
        rating = 0
    if user in userMap:
        user_avg_star =userMap[user][0]
        user_review_count = userMap[user][1]
    else:
        user_avg_star = 2.5
        user_review_count = 10 ##tune it correctly

    if business in businessMap:
        business_star = businessMap[business][0]
        business_review_count = businessMap[business][1]
    else:
        business_star =2.5
        business_review_count = 10 ## tune it correctly

    return [user,business,user_avg_star,user_review_count,business_star,business_review_count,rating]


st = time.time()
sc = SparkContext('local[*]','count_reviews')
sc.setLogLevel("ERROR")

path = sys.argv[1]
ip_business = sc.textFile(path+"/business.json")
ip_user = sc.textFile(path+"/user.json")

businessMap= dict(ip_business.map(loadBussinessJson).collect())

userMap = dict(ip_user.map(loadUserJson).collect())


def getModelBasedRatings():

    read_data= sc.textFile(path+"yelp_train.csv")
    first = read_data.first()
    read_data = read_data.filter(lambda x: x!= first)
    data = read_data.map(lambda x : createFeature(x)).collect()

    test_data= sc.textFile(path+sys.argv[2])
    first = test_data.first()
    test_data = test_data.filter(lambda x: x!= first)
    test_data = test_data.map(lambda x : createFeature(x)).collect()

    np_data = np.array(data)
    test_np_data = np.array(test_data)

    train_x = np_data[:,2:6].astype('float64')
    train_y = np_data[:,6].astype('float64')
    test_x = test_np_data[:,2:6].astype('float64')
    test_y = test_np_data[:,6].astype('float64')
    test_cases = test_np_data[:, 0:2]
    print("hiii")

    model = xgbr.XGBRegressor()
    model.fit(train_x,train_y)
    print(model)
    output = model.predict(data=test_x)

    rmseError = np.sqrt(np.mean((output-test_y)**2))
    predicted_rating = {}
    for A, B in zip(test_cases, output):
        predicted_rating[A[0] + "*" + A[1]] = B
    print()
    return predicted_rating


op = sys.argv[3]
op_file = open(op,"w")
op_file.write("user_id ,business_id ,prediction\n")
pr = getModelBasedRatings()
for i in pr:
    u = i.split("*")[0]
    b = i.split("*")[1]
    op_file.write(u+","+b+","+str(pr[i])+"\n")