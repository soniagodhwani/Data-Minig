from pyspark import SparkContext
import time
import pyspark
import sys
import csv
from itertools import combinations, chain

#<filter threshold> <support> <input_file_path> <output_file_path>
s = int(sys.argv[2])


def create_single_freq(baskets,threshhold):
    count_single = {}
    for basket in baskets:
        for item in basket:
            count_single[item] = count_single.get(item, 0) + 1

    freq = set()
    for i in count_single:
        if count_single[i] >= threshhold:
            freq.add(frozenset([i]))

    return freq



def get_next_freq(size, baskets, threshold,prev_freq):
    candidates = frozenset([frozenset(i.union(j)) for i in prev_freq for j in prev_freq if len(i.union(j)) == size])
    filtered = {}
    for i in baskets:
        for c in candidates:
            if c.issubset(i):
                filtered[c] = filtered.get(c,0)+1

    freq = set()
    for i in filtered:
        if filtered[i] >= threshold:
            freq.add( i)

    return freq


def pcy(index,iterator,threshhold,fth):
    #baskets = iterator
    baskets = []

    #print(fth)
    uBaskets = list(iter(iterator))
    for i in uBaskets:
        l = set(iter(i[1]))
        baskets.append(l)

        # if len(l) >= fth:
        #     baskets.append(l)


    global_freq = create_single_freq(baskets,threshhold)
    #print(index)
    #print(baskets)
    #print(global_freq)
    size = 2
    next = get_next_freq(size,baskets,threshhold,global_freq)
    while len(next) != 0:
        global_freq = global_freq.union(next)
        size += 1
        next = get_next_freq(size, baskets, threshhold, next)

    opt = []
    for i in global_freq:
        opt.append((tuple(sorted(i)),1))

    return opt

def count(iterator,candidates):
    baskets = []
    uBaskets = list(iter(iterator))
    for i in uBaskets:
        baskets.append(set(iter(i[1])))

    candidates_count = {}
    for i in baskets:
        for c in candidates:
            if set(c[0]).issubset(i):
                candidates_count[tuple(c[0])] = candidates_count.get(tuple(c[0]),0)+1

    counts = [(k, v) for k, v in candidates_count.items()]
    #print(candidates_count)
    return counts


def parse_csv_line(line):
    its = list(csv.reader(line))
    return (its[0][0]+"_"+str(int(its[2][0])),str(int(its[10][0])))
    #return line





# baskets = {}
# f = open(input_file,"r")
# rows = csv.reader(f)
# for i in rows:
#     if i[0] in baskets:
#         baskets[i[0]].append(i[1])
#     else:
#         baskets[i[0]] =[i[1]]
#
# print(pcy(0,baskets.values(),5))






# p = rdd1.getNumPartitions()
# candidates = rdd1.mapPartitionsWithIndex(lambda i,iterator: pcy(i,iterator,(1/p)*s)).reduceByKey(lambda x,y: x).collect()
# print(candidates)
# print(rdd1.mapPartitions(lambda iterator: count(iterator,candidates)).reduceByKey(lambda a,b: a+b).filter(lambda pair: pair[1]>=s).collect())

if __name__ == "__main__":
    st = time.time()
    outputDict = {}
    sc = SparkContext('local[*]', 'count_reviews')
    sc.setLogLevel("ERROR")
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    ip = sc.textFile(input_file)
    h = ip.first()

    ip = ip.filter(lambda x: x != h)

    fth = int(sys.argv[1])

    rdd1 = ip.map(parse_csv_line).groupByKey().filter(lambda x: len(x[1])>fth).partitionBy(None,hash).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    p = rdd1.getNumPartitions()
    op_file = open(output_file,"w")

    #print(fth)
    candidates = rdd1.mapPartitionsWithIndex(lambda i, iterator: pcy(i, iterator, (1 / p) * s,fth)).reduceByKey(lambda x, y: x).collect()
    print(len(candidates))
    candidates_op_map = {}
    for i in candidates:
        x = len(i[0])
        l = tuple(sorted(i[0]))
        if x in candidates_op_map:
            candidates_op_map[x].append(l)
        else:
            candidates_op_map[x] = [l]

    keys = sorted(candidates_op_map.keys())
    op_file.write("Candidates:\n")
    for i in keys:
        l = sorted(candidates_op_map[i])
        if i == 1:
            op_file.write((','.join('({})'.format("'" + t[0]+"'") for t in l)))
            op_file.write("\n")
            op_file.write("\n")
        else:
            op_file.write((','.join('{}'.format(t) for t in l)))
            op_file.write("\n\n")

    final_opt = rdd1.mapPartitions(lambda iterator: count(iterator, candidates)).reduceByKey(lambda a, b: a + b).filter(lambda pair: pair[1] >= s).collect()
    print(len(final_opt))
    final_opt_map = {}
    for i in final_opt:
        x = len(i[0])
        l = tuple(sorted(i[0]))
        if x in final_opt_map:

            final_opt_map[x].append(l)
        else:
            final_opt_map[x] = [l]

    keys = sorted(final_opt_map.keys())
    op_file.write("Frequent Itemsets:\n")
    for i in keys:
        l = sorted(final_opt_map[i])
        if i == 1:
            op_file.write((','.join('({})'.format("'" + t[0]+"'") for t in l)))
            op_file.write("\n")
            op_file.write("\n")
        else:
            op_file.write((','.join('{}'.format(t) for t in l)))
            op_file.write("\n\n")

    print("Duration: " + str(time.time() - st))
