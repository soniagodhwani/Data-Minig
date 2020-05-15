from pyspark import SparkContext
import math
import time
import sys

st = time.time()

sc = SparkContext('local[*]', 'LSH')

def parse(x):
    y = x.split(',')
    return (y[0],(y[1],y[2]))

def parse_test(x):
    y = x.split(',')
    return (y[0],y[1],y[2])

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
        return 0.5
    return  float(num)/den



def predictRating(user_id, business_id,usersToBusinessMap,businessesToUserMap,avgRatingsMap,weightsMap):
    if (user_id not in usersToBusinessMap) and (business_id not in businessesToUserMap):
        predictedRatingX = 3
    elif user_id not in usersToBusinessMap:
        predictedRatingX = avgRatingsMap[business_id]
    elif business_id not in businessesToUserMap:
        ratingTups = list(iter(usersToBusinessMap[user_id]))
        predictedRatingX = sum([i[1] for i in ratingTups])/float(len(ratingTups))
    elif business_id in dict(iter(usersToBusinessMap[user_id])):
        return dict(iter(usersToBusinessMap[user_id]))[business_id]
    else:
        predictedRatingX = getPearsonRating(user_id,business_id,usersToBusinessMap,businessesToUserMap,avgRatingsMap,weightsMap)
        #print(predictedRatingX)

    #predictedRating[user_id+"*"+business_id] = predictedRatingX
    return (user_id+"*"+business_id,predictedRatingX)


def getAvgOfNonzeroes(ratings):
    ratings = list(iter(ratings))
    num_sum = 0
    for i in ratings:
        num_sum += float(i[1])
    return num_sum/len(ratings)


def getItemBasedCFRating():
    read_data= sc.textFile(sys.argv[1])
    rdd_data= read_data.map(lambda x : parse(x))
    rdd= rdd_data.filter(lambda x: x[0]!= "user_id").persist()

    test_data= sc.textFile(sys.argv[2])
    test_data= test_data.map(lambda x : parse_test(x))
    test= test_data.filter(lambda x: x[0]!= "user_id").persist()

    disticnt_users = rdd.map(lambda x:x[0]).distinct().collect()
    disticnt_businesses = rdd.map(lambda x:x[1][0]).distinct().collect()

    print(len(disticnt_users))
    print(len(disticnt_businesses))

    usersToBusiness_rdd= rdd.map(lambda a:(a[0],(a[1][0],float(a[1][1])))).groupByKey().persist()
    usersToBusinessMap = usersToBusiness_rdd.collectAsMap()

    businessesToUser_rdd= rdd.map(lambda a:(a[1][0],(a[0],float(a[1][1])))).groupByKey().persist()

    avgRatings = businessesToUser_rdd.mapValues(lambda x: getAvgOfNonzeroes(x)).persist()

    businessesToUserMap = businessesToUser_rdd.collectAsMap()

    avgRatingsMap = avgRatings.collectAsMap()



    weightsMap = {}

    predictedRating = test.map(lambda x: predictRating(x[0],x[1],usersToBusinessMap,businessesToUserMap,avgRatingsMap,weightsMap))

    predictedRating = predictedRating.collect()
    predictedRating = dict(predictedRating)
    actual_ratings = dict(test.map(lambda x: (("*").join([x[0],x[1]]),x[2])).collect())

    num = 0
    den = len(predictedRating)
    for i in predictedRating:
        err = math.pow((predictedRating[i] - float(actual_ratings[i])),2)
        num += err

    rmse = math.sqrt(num/den)


    print("rmse " +str(rmse))
    print(len(test.collect()))
    print(time.time() - st)
    return predictedRating


op = sys.argv[3]
op_file = open(op,"w")
op_file.write("user_id,business_id,prediction\n")
pr = getItemBasedCFRating()

for i in pr:
    u = i.split("*")[0]
    b = i.split("*")[1]
    op_file.write(u+","+b+","+str(pr[i])+"\n")