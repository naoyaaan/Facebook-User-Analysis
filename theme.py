import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import community
import random
from gensim.models import Word2Vec as word2vec

def make_random_walks(G, num_of_walk, length_of_walk):
  walks = list()
  for i in range(num_of_walk):
    node_list = list(G.nodes())
    for node in node_list:
      current_node = node
      walk = list()
      walk.append(str(node))
      for j in range(length_of_walk):
        next_node = random.choice(list(G.neighbors(current_node)))
        walk.append(str(next_node))
        current_node = next_node
      walks.append(walk)
  return walks

def draw_h(G, pos, measures, measure_name):
    plt.figure(figsize=(10, 10))    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=10, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=list(measures.keys()))
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos)
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()

# Read facebook data
G = nx.read_edgelist('facebook_combined.txt', nodetype=int)
print(nx.info(G))

# visualiz this network
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_nodes(G, pos,node_size=10)


# visualize the centrality
draw_h(G, pos, nx.degree_centrality(G), 'Degree Centrality')
draw_h(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality')


# bfs : check the six degree of separation
d1 = list(nx.bfs_edges(G, source=0, depth_limit=1))
d2 = list(nx.bfs_edges(G, source=0, depth_limit=2))
d3 = list(nx.bfs_edges(G, source=0, depth_limit=3))
d4 = list(nx.bfs_edges(G, source=0, depth_limit=4))
d5 = list(nx.bfs_edges(G, source=0, depth_limit=5))
d6 = list(nx.bfs_edges(G, source=0, depth_limit=6))
d7 = list(nx.bfs_edges(G, source=0, depth_limit=7))
d8 = list(nx.bfs_edges(G, source=0, depth_limit=8))
d9 = list(nx.bfs_edges(G, source=0, depth_limit=9))
print("depth 7:", list(set(d7)-set(d6)))
print("depth 8:", list(set(d8)-set(d7)))
print("depth 9:", list(set(d9)-set(d8)))


# communities separation
partition = community.best_partition(G)
plt.figure(figsize=(10, 10))
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size=10, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.show(G)

# link prediction based on network embedding
walks = make_random_walks(G, 100, 20)
model = word2vec(walks, min_count=0, size=5, window=5, workers=1)

vlist = list()
for node in G.nodes():
  vector = model.wv[str(node)]
  vlist.append(vector)

DW = []
k = 5
n = nx.number_of_nodes(G)
for x in range(n):
  for y in range(x+1, n):
    if not(G.has_edge(x, y)):
      DW.append(tuple([x, y, np.linalg.norm(vlist[x]-vlist[y])]))
print("link prediction based on network embedding")
print(sorted(DW, key=lambda x:x[2], reverse=False)[:k])