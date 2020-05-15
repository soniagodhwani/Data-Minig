import random
import time
from sklearn.cluster import KMeans
from pyspark import SparkContext
import numpy as np
import math
from sklearn.metrics import normalized_mutual_info_score
import sys

random.seed(553)
RS = []
RS_ids = []
NUM_CLUSTERS = int(sys.argv[2])
DS_clusterStats = {}
DS_clusterIdToPointIDs = {}
CS_clusterStats = {}
CS_clusterIdToPointIDs = {}
output = []


def create_pointwise_clusterid(cluster):
    opt = []
    for pointID in cluster[1]:
        opt.append((pointID, cluster[0]))
    return opt

def getOptStats(index):
    global DS_clusterStats,CS_clusterStats,RS_ids
    n1 = 0
    for i in DS_clusterStats:
        n1 += DS_clusterStats[i][0]
    n2 = len(CS_clusterStats)
    n3 = 0
    for i in CS_clusterStats:
        n3 += CS_clusterStats[i][0]
    n4 = len(RS)

    return [index,n1,n2,n3,n4]

def getInterClusterDistance(c1, c2):
    global CS_clusterStats
    clus1 = CS_clusterStats[c1]
    clus2 = CS_clusterStats[c2]
    centoid_diff = (clus1[1] / clus1[0]) - (clus2[1] / clus2[0])
    dev1 = np.sqrt(clus1[2] / clus1[0] - ((clus1[1] / clus1[0]) ** 2))
    dev2 = np.sqrt(clus2[2] / clus2[0] - ((clus2[1] / clus2[0]) ** 2))
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for d in range(0, len(centoid_diff)):
        sum0 += ((centoid_diff[d]) ** 2) / (dev1[d] * dev2[d])
        sum1 += ((centoid_diff[d]) / dev2[d]) ** 2
        sum2 += ((centoid_diff[d]) / dev1[d]) ** 2

    return math.sqrt(min(sum1,sum2))


def mergeClusters(c1, c2):
    global CS_clusterStats,CS_clusterIdToPointIDs
    print("merging CS")
    clus1 = CS_clusterStats[c1]
    clus2 = CS_clusterStats[c2]

    N = clus1[0] + clus2[0]
    SUMN = clus1[1] + clus2[1]
    SUMNQ = clus1[2] + clus2[2]

    CS_clusterStats[c1] = [N, SUMN, SUMNQ]
    CS_clusterIdToPointIDs[c1].extend(CS_clusterIdToPointIDs[c2])


def mregeCSClusters(threshold):
    global CS_clusterStats, RS,CS_clusterIdToPointIDs
    flag = True
    while flag:
        flag = False
        keys1 = sorted(CS_clusterStats.keys())
        keys2 = sorted(CS_clusterStats.keys())
        for i in range(0, len(keys1) - 1):
            delete_i = []
            for j in range(i + 1, len(keys2)):
                if keys1[i] in CS_clusterStats and keys2[j] in CS_clusterStats:
                    d = getInterClusterDistance(keys1[i], keys2[j])
                    if d < threshold:
                        flag = True
                        mergeClusters(keys1[i], keys2[j])
                        delete_i.append(keys2[j])

            for del_clus in delete_i:
                del CS_clusterStats[del_clus]
                del CS_clusterIdToPointIDs[del_clus]


def create_CSclusters():
    global RS,RS_ids, NUM_CLUSTERS, CS_clusterStats,CS_clusterIdToPointIDs
    if len(RS) > 5 * NUM_CLUSTERS:
        # RS_data = np.array(RS)
        kmeans = KMeans(n_clusters=5 * NUM_CLUSTERS, random_state=0)
        point_cluster_arr = kmeans.fit_predict(RS)
        delete_indexes = []
        clusters = sc.parallelize(point_cluster_arr).zipWithIndex().map(lambda x: (x[0], [x[1]])).reduceByKey(
            lambda x, y: x + y).collect()
        # cluster => [clust_num, clust_point_ids]
        for cluster in clusters:
            if len(cluster[1]) != 1:
                cluster_points = [RS[i] for i in cluster[1]]
                cluster_pointIDs = [RS_ids[i] for i in cluster[1]]
                N = len(cluster[1])
                SUMN = np.sum(cluster_points, axis=0)
                SUMSQN = np.sum(np.array(cluster_points) ** 2, axis=0)
                key = 20
                while key in CS_clusterStats.keys():
                    key = random.randint(20, 1000)
                CS_clusterStats[key] = [N, SUMN, SUMSQN]
                CS_clusterIdToPointIDs[key] = cluster_pointIDs
                delete_indexes.extend(cluster[1])
        RS = np.delete(RS, delete_indexes, axis=0).tolist()
        RS_ids = np.delete(RS_ids, delete_indexes, axis=0).tolist()


def updateStats(point, type, clus,pointId):
    global RS,RS_ids, DS_clusterStats, DS_clusterIdToPointIDs, CS_clusterStats, CS_clusterIdToPointIDs
    if type == "DS":
        DS_clusterIdToPointIDs[clus].append(pointId)
        DS_clusterStats[clus][0] = DS_clusterStats[clus][0] + 1
        DS_clusterStats[clus][1] = DS_clusterStats[clus][1] + np.array(point)
        DS_clusterStats[clus][2] = DS_clusterStats[clus][2] + np.array(point) ** 2

    elif type == "CS":
        CS_clusterIdToPointIDs[clus].append(pointId)
        CS_clusterStats[clus][0] = CS_clusterStats[clus][0] + 1
        CS_clusterStats[clus][1] = CS_clusterStats[clus][1] + np.array(point)
        CS_clusterStats[clus][2] = CS_clusterStats[clus][2] + np.array(point) ** 2

    else:
        RS.append(point)
        RS_ids.append(pointId)


def get20Data(inp_lines, index):
    return np.array_split(inp_lines, 5)[index]


def getMahalanobisDistance(pointDim, clus):
    N = clus[0]
    SUMN = clus[1]
    SUMSQN = clus[2]
    centroid = SUMN / N

    dev = np.sqrt((SUMSQN / N - ((SUMN / N) ** 2)).tolist())
    sum = 0
    for d in range(0, len(pointDim)):
        sum += ((pointDim[d] - centroid[d]) / dev[d]) ** 2

    return math.sqrt(sum)

def getAssignment(pointDim):
    global DS_clusterStats, CS_clusterStats, RS
    threshold = 2 * math.sqrt(len(pointDim))
    min = threshold + 1
    min_clus = None
    for DS_clus in DS_clusterStats:  ## !!STEP8!!##
        mahaDistance = getMahalanobisDistance(pointDim, DS_clusterStats[DS_clus])
        if mahaDistance < min:
            min = mahaDistance
            min_clus = DS_clus
            return ("DS", min_clus)
    if min < threshold:
        return ("DS", min_clus)
    min = threshold + 1
    for CS_clus in CS_clusterStats:  ## !!STEP9!!##
        mahaDistance = getMahalanobisDistance(pointDim, CS_clusterStats[CS_clus])
        if mahaDistance < min:
            min = mahaDistance
            min_clus = CS_clus
            return ("CS", min_clus)
    if min < threshold:
        return ("CS", min_clus)
    return ("RS", None)  ## !!STEP10!!##

if __name__ == '__main__':
    st = time.time()
    sc = SparkContext()
    sc.setLogLevel("ERROR")

    ## number of clusters in input = 11
    inp_filename = sys.argv[1]
    opt_filename = sys.argv[3]
    opt_file = open(opt_filename,"w")
    inp_file = open(inp_filename, "r")
    inp_lines = inp_file.readlines()

    print("total lines: "+str(len(inp_lines)))

    random.shuffle(inp_lines)
    initial_data = get20Data(inp_lines, 0)  ## !!STEP1!!##
    print("0: " +str(len(initial_data)))

    dataRdd = sc.parallelize(initial_data)

    data = dataRdd.map(lambda x: [float(y) for y in x.split(",")[2:]]).collect()
    data_ids = dataRdd.map(lambda x: int(x.split(",")[0])).collect()


    # ========= Create Initial data for clustering
    np_array = np.array(data)

    kmeans = KMeans(n_clusters=5 * NUM_CLUSTERS, random_state=0)  ## !!STEP2!!##
    point_cluster_arr = kmeans.fit_predict(np_array)
    cluster_centers = kmeans.cluster_centers_

    # ===== Create RS points and Points to remove from Original data

    delete_indexes = []
    RS_indexes = sc.parallelize(point_cluster_arr).zipWithIndex().map(lambda x: (x[0], [x[1]])).reduceByKey(
        lambda x, y: x + y).filter(lambda x: len(x[1]) == 1).collect()  ## !!STEP3!!##
    for i in RS_indexes:
        point = np_array[i[1][0]]
        RS.append(point)
        RS_ids.append(data_ids[i[1][0]])
        delete_indexes.append(i[1][0])

    print(len(RS))

    # ====create new data for clustering
    new_data = np.delete(np_array, delete_indexes, 0)
    new_data_ids = np.delete(data_ids, delete_indexes, 0)

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)  ## !!STEP4!!##
    point_cluster_arr = kmeans.fit_predict(new_data)
    cluster_centers = kmeans.cluster_centers_

    # ======= Generate DS Clusters
    point_cluster_tuple_arr = np.column_stack((point_cluster_arr.astype(np.object), new_data))
    pointid_cluster_tuple_arr = np.column_stack((point_cluster_arr.astype(np.object), new_data_ids))
    DS_clusterIdToPoint1 = sc.parallelize(point_cluster_tuple_arr).map(lambda x: (int(x[0]), [x[1:]])).reduceByKey(
        lambda x, y: x + y).collectAsMap()
    DS_clusterIdToPointIDs = sc.parallelize(pointid_cluster_tuple_arr).map(lambda x: (int(x[0]), [x[1]])).reduceByKey(
        lambda x, y: x + y).collectAsMap()


    ## !!STEP5!!##
    for cluster in DS_clusterIdToPoint1:
        N = len(DS_clusterIdToPoint1[cluster])
        SUMN = np.sum(DS_clusterIdToPoint1[cluster],axis=0)
        SUMSQN = np.sum(np.array(DS_clusterIdToPoint1[cluster]) ** 2, axis=0)
        DS_clusterStats[cluster] = [N, SUMN, SUMSQN]

    # ======= Run K-Means on RS points

    ## !!STEP6!!##
    create_CSclusters()

    index = 1

    output.append(getOptStats(index))

    while index < 5:
        data = get20Data(inp_lines, index)  ## !!STEP7!!##
        print(str(index) + ": " + str(len(data)))
        for point in data:
            point_details = point.split(",")
            pointDim = [float(y) for y in point_details[2:]]
            pointId = int(point_details[0])
            assignment = getAssignment(pointDim)
            updateStats(pointDim, assignment[0], assignment[1],pointId)
        create_CSclusters()  ## STEP11##
        t = 2 * math.sqrt(len(pointDim))
        mregeCSClusters(t)  ## STEP12##
        if index == 4:
            del_cluss = []
            for clus in CS_clusterStats:
                centroid = CS_clusterStats[clus][1] / CS_clusterStats[clus][0]
                assignment = getAssignment(centroid)
                if assignment[0] == "DS":
                    DS_clusterStats[assignment[1]][0] = DS_clusterStats[assignment[1]][0] + CS_clusterStats[clus][0]
                    DS_clusterStats[assignment[1]][1] = DS_clusterStats[assignment[1]][1] + CS_clusterStats[clus][1]
                    DS_clusterStats[assignment[1]][2] = DS_clusterStats[assignment[1]][2] + CS_clusterStats[clus][2]
                    DS_clusterIdToPointIDs[assignment[1]].extend(CS_clusterIdToPointIDs[clus])
                    del_cluss.append(clus)

            for del_clus in del_cluss:
                del CS_clusterStats[del_clus]
                del CS_clusterIdToPointIDs[del_clus]

        output.append(getOptStats(index+1))
        index += 1

    for clus in CS_clusterIdToPointIDs:
        RS_ids.extend(CS_clusterIdToPointIDs[clus])

    DS_clusterIdToPointIDs[-1] = RS_ids


    pointID_to_clusterID_rdd = sc.parallelize(DS_clusterIdToPointIDs.items()).flatMap(create_pointwise_clusterid).sortBy(lambda x: x[0])
    pointID_to_clusterID = pointID_to_clusterID_rdd.collect()
    cluster_by_id  = pointID_to_clusterID_rdd.map(lambda x : x[1]).collect()

    inp_file = open(inp_filename, "r")
    inp_lines = inp_file.readlines()

    actual_pointIndex_cluster = sc.parallelize(inp_lines).map(lambda x : int(x.split(",")[1])).collect()

    x = normalized_mutual_info_score(actual_pointIndex_cluster, cluster_by_id)
    print(x)

    opt_file.write("The intermediate results:\n")
    for i in range(len(output)):
        opt_file.write("Round " + str(output[i][0] ) + ": " + str(output[i][1]) + "," + str(output[i][2]) + "," + str(
            output[i][3]) + "," + str(output[i][4]) + "\n")

    opt_file.write("\nThe clustering results:\n")

    for i in pointID_to_clusterID:
        opt_file.write(str(i[0]) + "," + str(i[1]) + "\n")

    print(time.time() - st)