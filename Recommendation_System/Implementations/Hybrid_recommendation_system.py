import json
from pyspark import SparkContext
import time
import numpy as np
import xgboost as xgbr
from sklearn.metrics import mean_squared_error
import math
import sys


def loadBussinessJson(line):
    business = json.loads(line)
    return (business['business_id'], [float(business['stars']), float(business['review_count'])])


def loadUserJson(line):
    user = json.loads(line)
    return (user['user_id'], [float(user['average_stars']), float(user['review_count'])])


def createFeature(input):
    ip_data = input.split(',')
    user = ip_data[0]
    business = ip_data[1]
    if len(ip_data) == 3:
        rating = float(ip_data[2])
    else:
        rating = 0
    if user in userMap:
        user_avg_star = userMap[user][0]
        user_review_count = userMap[user][1]
    else:
        user_avg_star = 2.5
        user_review_count = 10  ##tune it correctly

    if business in businessMap:
        business_star = businessMap[business][0]
        business_review_count = businessMap[business][1]
    else:
        business_star = 2.5
        business_review_count = 10  ## tune it correctly

    return [user, business, user_avg_star, user_review_count, business_star, business_review_count, rating]


st = time.time()
sc = SparkContext('local[*]', 'count_reviews')
sc.setLogLevel("ERROR")

path = sys.argv[1]
ip_business = sc.textFile(path + "/business.json")
ip_user = sc.textFile(path + "/user.json")

businessMap = dict(ip_business.map(loadBussinessJson).collect())

userMap = dict(ip_user.map(loadUserJson).collect())


def getModelBasedRatings():
    read_data = sc.textFile("yelp_train.csv")
    first = read_data.first()
    read_data = read_data.filter(lambda x: x != first)
    data = read_data.map(lambda x: createFeature(x)).collect()

    test_data = sc.textFile(sys.argv[2])
    first = test_data.first()
    test_data = test_data.filter(lambda x: x != first)
    test_data = test_data.map(lambda x: createFeature(x)).collect()

    np_data = np.array(data)
    test_np_data = np.array(test_data)

    train_x = np_data[:, 2:6].astype('float64')
    train_y = np_data[:, 6].astype('float64')
    test_x = test_np_data[:, 2:6].astype('float64')
    test_y = test_np_data[:, 6].astype('float64')
    test_cases = test_np_data[:,0:2]
    #print("hiii")

    model = xgbr.XGBRegressor()
    model.fit(train_x, train_y)
    #print(model)
    output = model.predict(data=test_x)

    #rmseError = np.sqrt(np.mean((output - test_y) ** 2))
    # rmseError = mean_squared_error(test_y, output)
    #print('RMSE Error is : ', rmseError)
    #print(time.time() - st)

    predicted_rating = {}
    #actual_rating = {}
    for A, B in zip(test_cases, output):
        predicted_rating[A[0]+"*"+A[1]] = float(B)


    # for A, B in zip(test_cases, test_y):
    #     actual_rating[A[0] + "*" + A[1]] = float(B)


    return predicted_rating


def parse(x):
    y = x.split(',')
    return (y[0], (y[1], y[2]))


def parse_test(x):
    y = x.split(',')
    return (y[0], y[1],y[2])


def getPearsonRating(user_id,business_id,usersToBusinessMap,businessesToUserMap,avgRatingsMap,weightsMap):
    ratingTups = list(iter(usersToBusinessMap[user_id]))
    corated_businesses = [i[0] for i in ratingTups]
    for business in corated_businesses:
        if ("*").join(sorted([business,business_id])) not in weightsMap:
            item1 = dict(iter(businessesToUserMap[business_id]))
            item2 = dict(iter(businessesToUserMap[business]))
            avg1 = avgRatingsMap[business_id]
            avg2 = avgRatingsMap[business]
            num = 0
            den1 = 0
            den2 = 0
            for user in item1:
                if user in item2:
                    num += (item1[user] - avg1)*(item2[user] - avg2)
                    den1 += math.pow((item1[user] - avg1),2)
                    den2 += math.pow((item2[user] - avg2),2)
            if den1 == 0 or den2 == 0:
                w = 1 - (abs(avg1 - avg2) / 2)
                if w< 0:
                    w =0
                if w > 1:
                    w = 1
                weightsMap[("*").join(sorted([business, business_id]))] = w
            else:
                w = float(num)/(math.sqrt(den1) * math.sqrt(den2))
                if w< 0:
                    w =0
                if w > 1:
                    w = 1
                weightsMap[("*").join(sorted([business,business_id]))] = w
    num = 0
    den = 0
    for i in ratingTups:
        w = weightsMap[("*").join(sorted([i[0],business_id]))]
        num += i[1]* w
        den += abs(w)
    if den == 0:
        return 0
    return  float(num)/den


def predictRating(user_id, business_id, usersToBusinessMap, businessesToUserMap, avgRatingsMap, weightsMap):
    if (user_id not in usersToBusinessMap) and (business_id not in businessesToUserMap):
        predictedRatingX = 2.5
    elif user_id not in usersToBusinessMap:
        predictedRatingX = avgRatingsMap[business_id]
    elif business_id not in businessesToUserMap:
        ratingTups = list(iter(usersToBusinessMap[user_id]))
        predictedRatingX = sum([i[1] for i in ratingTups]) / float(len(ratingTups))
    elif business_id in dict(iter(usersToBusinessMap[user_id])):
        return dict(iter(usersToBusinessMap[user_id]))[business_id]
    else:
        predictedRatingX = getPearsonRating(user_id, business_id, usersToBusinessMap, businessesToUserMap,
                                            avgRatingsMap, weightsMap)
        # print(predictedRatingX)

    # predictedRating[user_id+"*"+business_id] = predictedRatingX
    return (user_id + "*" + business_id, predictedRatingX)


def getAvgOfNonzeroes(ratings):
    ratings = list(iter(ratings))
    num_sum = 0
    for i in ratings:
        num_sum += float(i[1])
    return num_sum / len(ratings)



read_data = sc.textFile(path+ "yelp_train.csv")
rdd_data = read_data.map(lambda x: parse(x))
rdd = rdd_data.filter(lambda x: x[0] != "user_id").persist()

test_data = sc.textFile(sys.argv[2])
test_data = test_data.map(lambda x: parse_test(x))
test = test_data.filter(lambda x: x[0] != "user_id").persist()

usersToBusiness_rdd = rdd.map(lambda a: (a[0], (a[1][0], float(a[1][1])))).groupByKey().persist()
usersToBusinessMap = usersToBusiness_rdd.collectAsMap()

businessesToUser_rdd = rdd.map(lambda a: (a[1][0], (a[0], float(a[1][1])))).groupByKey().persist()

avgRatings = businessesToUser_rdd.mapValues(lambda x: getAvgOfNonzeroes(x)).persist()

businessesToUserMap = businessesToUser_rdd.collectAsMap()

print("businessesToUserMap:")
print(len(businessesToUserMap))

print("usersToBusinessMap")
print(len(usersToBusinessMap))

avgRatingsMap = avgRatings.collectAsMap()

weightsMap = {}

predictedRating = test.map(
    lambda x: predictRating(x[0], x[1], usersToBusinessMap, businessesToUserMap, avgRatingsMap, weightsMap))

predictedRating = predictedRating.collect()
predictedRating = dict(predictedRating)
actual_ratings = dict(test.map(lambda x: (("*").join([x[0], x[1]]), x[2])).collect())

num = 0
den = len(predictedRating)
for i in predictedRating:
    err = math.pow((predictedRating[i] - float(actual_ratings[i])), 2)
    num += err

rmse = math.sqrt(num / den)

# print(rmse)
# print(len(test.collect()))
# print(time.time() - st)


pr1 = predictedRating
model_based = getModelBasedRatings()
pr2 = model_based

#
# num =0
# den = len(actual_ratings)
f1 = open("test.csv","w")
f = open("users.txt","w")
pr = {}
pr_t = []
op=[]
for i in pr1:
    business = i.split("*")[1]
    user = i.split("*")[0]
    if business  in businessesToUserMap and user  in usersToBusinessMap:
        f.write(str(len(businessesToUserMap[business])) +"  "+ str(len(usersToBusinessMap[user]))+"\n")
    if business not in businessesToUserMap  or user not in usersToBusinessMap or len(businessesToUserMap[business]) < 250 or len(usersToBusinessMap[user]) < 250 :
        pr[i]= pr2[i]
    elif len(businessMap[business]) < 2500:
        print("hi 1")
        pr[i] = 0.05*pr1[i]+0.95*pr2[i]
    else:
        print("hi 2")
        pr[i] = 0.1 * pr1[i] + 0.9 * pr2[i]
    pr_t.append(pr[i])
    op.append(actual_ratings[i])



op = np.array(op).astype('float64')
pr_t = np.array(pr_t).astype('float64')
rmse = np.sqrt(np.mean((pr_t-op)**2))

op = sys.argv[3]
op_file = open(op,"w")
op_file.write("user_id ,business_id ,prediction\n")

for i in pr:
    u = i.split("*")[0]
    b = i.split("*")[1]
    op_file.write(u+","+b+","+str(pr[i])+"\n")

