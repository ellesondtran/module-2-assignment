# Reddit network analysis - INST414
# looking at subreddit connections based on shared users
# nodes = subreddits, edges = users who posted in both

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from collections import defaultdict

# load data
# dataset: https://www.kaggle.com/datasets/colemaclean/subreddit-interactions
# file was too large to upload directly so i sampled it first:
# head -5001 reddit_data.csv > reddit_sample.csv

df = pd.read_csv('reddit_sample.csv')
print("shape:", df.shape)
print("unique users:", df['username'].nunique())
print("unique subreddits:", df['subreddit'].nunique())
print("\ntop subreddits:")
print(df['subreddit'].value_counts().head(10))

# build edges
# two subreddits are connected if the same user posted in both
user_subs = df.groupby('username')['subreddit'].apply(list)
edge_weights = defaultdict(int)

for subs in user_subs:
    unique_subs = list(set(subs))
    for a, b in combinations(unique_subs, 2):
        edge = tuple(sorted([a, b]))
        edge_weights[edge] += 1

# filter to pairs with at least 2 shared users
filtered = {e: w for e, w in edge_weights.items() if w >= 2}
print("\nedges after filtering:", len(filtered))

# build graph
G = nx.Graph()
for (a, b), w in filtered.items():
    G.add_edge(a, b, weight=w)

print("nodes:", G.number_of_nodes())
print("edges:", G.number_of_edges())

# importance metrics
pagerank    = nx.pagerank(G, weight='weight')
betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
degree      = dict(G.degree(weight='weight'))

results = pd.DataFrame({
    'subreddit':       list(G.nodes()),
    'pagerank':        [round(pagerank[n], 4) for n in G.nodes()],
    'betweenness':     [round(betweenness[n], 4) for n in G.nodes()],
    'weighted_degree': [degree[n] for n in G.nodes()],
}).sort_values('pagerank', ascending=False).reset_index(drop=True)

print("\ntop 10 by pagerank:")
print(results.head(10).to_string(index=False))
print("\ntop 5 by betweenness:")
print(results.sort_values('betweenness', ascending=False).head(5)[['subreddit','betweenness']].to_string(index=False))

# color nodes by rough topic category
news_subs    = {'worldnews','news','nottheonion','todayilearned','UpliftingNews'}
fun_subs     = {'gifs','pics','funny','mildlyinteresting','OldSchoolCool','food','WTF','aww'}
general_subs = {'AskReddit','GetMotivated','casualiama','space'}

def get_color(n):
    if n in news_subs: return '#e05c5c'
    if n in fun_subs: return '#4e8df5'
    if n in general_subs: return '#e8a838'
    return '#aaaaaa'

# visualize top 30 nodes
top30 = [n for n, _ in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:30]]
H = G.subgraph(top30)
pos = nx.spring_layout(H, seed=7, k=2.5)

node_colors = [get_color(n) for n in H.nodes()]
node_sizes  = [pagerank[n] * 25000 for n in H.nodes()]
edge_widths = [H[u][v]['weight'] / 30 for u, v in H.edges()]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# network graph
nx.draw_networkx_edges(H, pos, ax=axes[0], width=edge_widths, alpha=0.3, edge_color='#999')
nx.draw_networkx_nodes(H, pos, ax=axes[0], node_color=node_colors, node_size=node_sizes, alpha=0.9)
nx.draw_networkx_labels(H, pos, ax=axes[0], font_size=7, font_weight='bold')
patches = [mpatches.Patch(color='#e05c5c', label='News'),
           mpatches.Patch(color='#4e8df5', label='Entertainment'),
           mpatches.Patch(color='#e8a838', label='General'),
           mpatches.Patch(color='#aaaaaa', label='Other')]
axes[0].legend(handles=patches, loc='lower left', fontsize=8)
axes[0].set_title('Top 30 Subreddits by PageRank\n(node size = PageRank, edges = shared users)', fontweight='bold')
axes[0].axis('off')

# pagerank bar chart
top10_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
labels, vals = zip(*top10_pr)
axes[1].barh(list(reversed(labels)), list(reversed(vals)),
             color=[get_color(l) for l in reversed(labels)], edgecolor='white')
axes[1].set_xlabel('PageRank Score')
axes[1].set_title('Top 10 Subreddits by PageRank', fontweight='bold')
axes[1].spines[['top','right','left']].set_visible(False)
axes[1].tick_params(left=False)

plt.tight_layout()
plt.savefig('reddit_network_real.png', dpi=150, bbox_inches='tight')
plt.show()
print("chart saved.")
