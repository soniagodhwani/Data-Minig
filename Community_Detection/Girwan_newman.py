from pyspark import SparkContext
from copy import deepcopy
import time
import sys


def bfs(root, graph):
    q = [root]
    visited = set()
    while len(q) != 0:
        node = q.pop(0)
        if node not in visited:
            visited.add(node)
            q.extend(graph[node])
    return visited

def bfs_al(root,al):
    parents = {}
    numPaths = {}
    q = [(root, 0)]
    parents[root] = [(-1, -1)]
    numPaths[root] = 1
    visited = []

    while len(q) != 0:
        node = q.pop(0)
        if node[0] not in visited:
            if node[0] not in al:
                visited.append(node[0])
            else:
                for i in al[node[0]]:
                    if i not in parents:
                        parents[i] = [node]
                        numPaths[i] = numPaths[node[0]]
                    elif parents[i][0][1] == node[1]:
                        parents[i].append(node)
                        numPaths[i] = numPaths[i] + numPaths[node[0]]
                    q.append((i, node[1] + 1))
                visited.append(node[0])

    return visited, parents, numPaths



def getBetweenessFromNode(node,adjacency_list):
    visited,parents,numPaths = bfs_al(node,adjacency_list)
    visited.reverse()
    edge_betweeness = {}
    node_credit = {vertex: 1 for vertex in visited}
    for vertex in visited[:-1]:
        for parent in parents[vertex]:
            edge = (min(vertex, parent[0]),max(vertex, parent[0]))
            credit = float(node_credit[vertex] * numPaths[parent[0]])/float(numPaths[vertex])
            if edge not in edge_betweeness.keys():
                edge_betweeness[edge] = 0
            edge_betweeness[edge] += credit
            node_credit[parent[0]] += credit

    #return edge_betweeness
    return [(k, v) for k, v in edge_betweeness.items()]


def createAdjacencyList(edges):
    adjacency_list = {}
    for i in edges:
        if i[0] in adjacency_list:
            adjacency_list[i[0]].add(i[1])
        else:
            adjacency_list[i[0]] = set([i[1]])
        if i[1] in adjacency_list:
            adjacency_list[i[1]].add(i[0])
        else:
            adjacency_list[i[1]] = set([i[0]])

    return adjacency_list



def findConnectedComponents(adj_list):
    seen = set()
    connected_components = []
    for i in adj_list:
        if i not in seen:
            visited= bfs(i,adj_list)
            connected_components.append(visited)
            seen = seen.union(visited)
    return connected_components




#                            int,  int->set,    int ->set
def findModContriOfComponent(m, adjacency_list,component):
        modularity = 0
        for i in component:
            for j in component:
                wt = 0
                if j in adjacency_list[i]:
                    wt = 1
                modularity += wt - (len(adjacency_list[i])*len(adjacency_list[j])/float(2*m))
        return [(1,modularity)]


st = time.time()

sc = SparkContext('local[*]', 'LSH')
sc.setLogLevel("ERROR")

ip_data  = sc.textFile(sys.argv[1])

nodes = ip_data.flatMap(lambda x: (x.split(" ")[0],x.split(" ")[1])).distinct().persist()

nodes_data  = nodes.collect()

edges = ip_data.map(lambda x: (x.split(" ")[0],x.split(" ")[1])).collect()

adjacency_list = {}

adjacency_list = createAdjacencyList(edges)




betweeness = nodes.flatMap(lambda node: getBetweenessFromNode(node,adjacency_list)).reduceByKey(lambda x,y: x+y).mapValues(lambda x: x/2).sortBy(lambda x: (-1 * x[1],x[0][0],x[0][1])).collect()

bet_file = open(sys.argv[2],"w")
for bet in betweeness:
    bet_file.write(str(bet[0])+", "+str(bet[1])+"\n")

max_betweeness = betweeness[0]
# print(max_betweeness)



modularity = -1
m = len(edges)
c = len(edges)
connected_components = findConnectedComponents(adjacency_list)
argMaxComponents = connected_components

# print(len(connected_components))
connected_components_rdd = sc.parallelize(connected_components)

adjacency_list_copy = deepcopy(adjacency_list)

while c!=1:
    # print(c)

    mod = connected_components_rdd.flatMap(lambda component: findModContriOfComponent(m,adjacency_list,component))


    mod = float(mod.reduceByKey(lambda x,y : x+y).collect()[0][1])/(2*m)
    if mod > modularity:
        modularity = mod
        argMaxComponents = deepcopy(connected_components)

    u = max_betweeness[0][0]
    v = max_betweeness[0][1]

    adjacency_list_copy[u].remove(v)
    adjacency_list_copy[v].remove(u)

    max_betweeness = nodes.flatMap(lambda node: getBetweenessFromNode(node, adjacency_list_copy)).reduceByKey(
        lambda x, y: x + y).mapValues(lambda x: x / 2).sortBy(lambda x: -1 * x[1]).take(1)[0]


    connected_components = findConnectedComponents(adjacency_list_copy)
    # print(len(connected_components))
    connected_components_rdd = sc.parallelize(connected_components)

    c  -= 1

components = []
for comp in argMaxComponents:
    com = list(comp)
    com.sort()
    components.append(com)

components.sort()
components.sort(key = len)

comp_file  = open(sys.argv[3],"w")

for comp in components:
    comp_file.write(str(comp)[1:-1]+"\n")
#
# print(modularity)
# print(len(argMaxComponents))
# print(argMaxComponents)

# adj = {0:[1,2],1:[0,2],2:[0,1,3],3:[2,4,5,6],4:[3,5],5:[3,4,6],6:[3,5]}
# m = 9
# cc = {0:[1,2],1:[0,2],2:[0,1]}
#
# print(findModContriOfComponent(m,adj,cc))

print(time.time() - st)


##wrong modularity calculation and very slow