import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def load_embeddings(path, max_words=500):
    embeddings = {} # {"word": np.array([...]), ...}

    with open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            if index > max_words:
                break
            split_line = line.strip().split()
            word = split_line[0]
            word_embedding=np.array(split_line[1:], dtype=np.float64)
            embeddings[word] = word_embedding

        return embeddings

glove_embeddings = load_embeddings("data/glove.2024.wikigiga.50d_small.txt")
words, vecs = glove_embeddings.keys(), np.array(list(glove_embeddings.values()))

# use PCA to lower dimensions down to 50, for speed
def lower_dimensions(vecs):
    if vecs.shape[1] > 50:
        pca = PCA(n_components=50)
        vecs = pca.fit_transform(vecs)

    return vecs
lower_dimensions(vecs)

# find similarity between words
S = cosine_similarity(vecs) # N x N similarity matrix: words x context
S = (S - S.mean()) / S.std() #normalizing, so similarity is between -1 and 1

#setting up particles

N = len(words) # of course limited by max_words, but good to have
pos = np.random.rand(N,2) * 2.0 # position (x,y)
vel = np.zeros_like(pos) # zero velocity for now
mass = np.ones(N) #figure that out later

# force simulation parameters
alpha=0.95
beta=0.2
dt=0.05
damping=0.99

## Force Computation
def compute_forces(pos, S, alpha, beta):
    F = np.zeros_like(pos)

    for i in range(N):
        diff = pos[i] - pos

        #attract/repel
        strength = -alpha * (S[i] - beta)
        F[i] = np.sum(strength[:, None] * diff, axis=0)
    return F

for step in range(1000):
    F = compute_forces(pos, S, alpha, beta)
    vel += F *dt
    pos += vel*dt
    vel *= damping

    pos = np.clip(pos, -10, 10) #soft boundary

    if step % 20 == 0:
        plt.clf()

        if step % 200 == 0:
            kmeans = KMeans(n_clusters=8, random_state=0, n_init=10)
            clusters = kmeans.fit_predict(vecs)

        plt.scatter(pos[:, 0], pos[:, 1], c=clusters, s=8, cmap='tab10', alpha=0.7)
        plt.title(f"Step {step}")
        plt.pause(0.01)