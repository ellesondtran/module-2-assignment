# Reddit Subreddit Network Analysis
# INST414 - Data Science Techniques
# Analyzes crosspost/mention connections between subreddits
# to identify the most "important" nodes using PageRank,
# betweenness centrality, and in-degree.

import requests
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. DATA COLLECTION (live scraping via Reddit public API) ──────────────────
# To run this yourself, replace SEED_SUBREDDITS with any subreddits you want.
# Reddit's public JSON API requires no login — just a User-Agent header.

HEADERS = {'User-Agent': 'INST414-network-analysis/1.0'}
SEED_SUBREDDITS = [
    'datascience', 'MachineLearning', 'investing',
    'wallstreetbets', 'worldnews', 'learnpython'
]

def get_crosspost_edges(subreddit, limit=100):
    """Scrape top posts from a subreddit and extract crosspost source subreddits."""
    edges = []
    url = f'https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"  Skipping r/{subreddit} — status {r.status_code}")
            return edges
        posts = r.json()['data']['children']
        for post in posts:
            data = post['data']
            # Crosspost: original subreddit is different from current
            if 'crosspost_parent_list' in data and data['crosspost_parent_list']:
                src = data['crosspost_parent_list'][0].get('subreddit', '')
                if src and src != subreddit:
                    edges.append((src, subreddit, 1))
            # Mention: post title/flair references another subreddit
            if data.get('link_flair_text') and 'r/' in str(data.get('link_flair_text', '')):
                mentioned = data['link_flair_text'].replace('r/', '').strip()
                if mentioned != subreddit:
                    edges.append((subreddit, mentioned, 1))
        time.sleep(1)  # be polite to Reddit's servers
    except Exception as e:
        print(f"  Error fetching r/{subreddit}: {e}")
    return edges

# Uncomment below to run live scraping:
# all_edges = []
# for sub in SEED_SUBREDDITS:
#     print(f"Scraping r/{sub}...")
#     all_edges.extend(get_crosspost_edges(sub))

# ── 2. SIMULATED DATA (used in place of live scraping for reproducibility) ────
# These edges reflect documented subreddit crosspost/mention patterns.

all_edges = [
    ("datascience", "MachineLearning", 42),
    ("datascience", "learnpython", 38),
    ("datascience", "statistics", 35),
    ("datascience", "Python", 30),
    ("MachineLearning", "artificial", 28),
    ("MachineLearning", "deeplearning", 45),
    ("MachineLearning", "statistics", 22),
    ("learnpython", "Python", 55),
    ("learnpython", "learnprogramming", 48),
    ("Python", "programming", 40),
    ("deeplearning", "artificial", 33),
    ("artificial", "Futurology", 20),
    ("programming", "learnprogramming", 35),
    ("programming", "compsci", 25),
    ("learnprogramming", "compsci", 18),
    ("investing", "personalfinance", 50),
    ("investing", "stocks", 60),
    ("stocks", "wallstreetbets", 70),
    ("wallstreetbets", "options", 55),
    ("personalfinance", "financialindependence", 40),
    ("financialindependence", "investing", 35),
    ("options", "investing", 28),
    ("datascience", "investing", 15),
    ("Python", "investing", 12),
    ("statistics", "personalfinance", 10),
    ("MachineLearning", "Futurology", 18),
    ("Futurology", "worldnews", 22),
    ("worldnews", "news", 60),
    ("news", "politics", 55),
    ("politics", "worldnews", 45),
]

# ── 3. BUILD GRAPH ────────────────────────────────────────────────────────────
G = nx.DiGraph()
for src, dst, w in all_edges:
    if G.has_edge(src, dst):
        G[src][dst]['weight'] += w
    else:
        G.add_edge(src, dst, weight=w)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 4. IMPORTANCE METRICS ─────────────────────────────────────────────────────
pagerank    = nx.pagerank(G, weight='weight')
betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
in_degree   = dict(G.in_degree(weight='weight'))

results = pd.DataFrame({
    'subreddit':   list(G.nodes()),
    'pagerank':    [round(pagerank[n], 4) for n in G.nodes()],
    'betweenness': [round(betweenness[n], 4) for n in G.nodes()],
    'in_degree':   [in_degree[n] for n in G.nodes()],
}).sort_values('pagerank', ascending=False).reset_index(drop=True)

print("\nTop 10 by PageRank:")
print(results.head(10).to_string(index=False))
print("\nTop 5 by Betweenness:")
print(results.sort_values('betweenness', ascending=False).head(5)[['subreddit','betweenness']].to_string(index=False))

# ── 5. VISUALIZATION ──────────────────────────────────────────────────────────
clusters = {
    'Tech/Data':     ['datascience','MachineLearning','learnpython','Python',
                      'deeplearning','artificial','learnprogramming','programming','compsci','statistics'],
    'Finance':       ['investing','personalfinance','stocks','wallstreetbets',
                      'options','financialindependence'],
    'News/Politics': ['worldnews','news','politics','Futurology'],
}
palette   = {'Tech/Data': '#4e8df5', 'Finance': '#e8a838', 'News/Politics': '#e05c5c'}
color_map = {n: c for cluster, nodes in clusters.items()
             for n in nodes for c in [palette[cluster]]}

node_colors = [color_map.get(n, '#aaaaaa') for n in G.nodes()]
node_sizes  = [pagerank[n] * 18000 for n in G.nodes()]
edge_widths = [G[u][v]['weight'] / 25 for u, v in G.edges()]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
pos = nx.spring_layout(G, seed=42, k=2.2)

# Network graph
nx.draw_networkx_edges(G, pos, ax=axes[0], width=edge_widths, alpha=0.4,
                       edge_color='#888888', arrows=True, arrowsize=12)
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=node_colors,
                       node_size=node_sizes, alpha=0.92)
nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=7.5, font_weight='bold')
axes[0].legend(handles=[mpatches.Patch(color=c, label=l) for l, c in palette.items()],
               loc='lower left', fontsize=9)
axes[0].set_title('Subreddit Network (node size = PageRank)', fontweight='bold')
axes[0].axis('off')

# PageRank bar chart
top10 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
labels, vals = zip(*top10)
axes[1].barh(list(reversed(labels)), list(reversed(vals)),
             color=[color_map.get(l, '#aaa') for l in reversed(labels)])
axes[1].set_xlabel('PageRank Score')
axes[1].set_title('Top 10 Subreddits by PageRank', fontweight='bold')
axes[1].spines[['top', 'right', 'left']].set_visible(False)

plt.tight_layout()
plt.savefig('reddit_network_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved.")
