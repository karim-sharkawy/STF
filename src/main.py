import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

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

glove_embeddings = load_embeddings("glove.6B.50d.txt")
words, vecs = glove_embeddings.keys(), glove_embeddings.values()

# use PCA to lower dimensions down to 50, for speed
def lower_dimensions(vecs):
    if vecs.shape[1] > 50:
        pca = PCA(n_components=50)
        vecs = pca.fit_transform(vecs)

    return vecs

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
damping=0.99 # will use later for velocity

## Force Computation
def compute_forces(pos, S, alpha, beta):
    F = np.zeros_like(pos)

    for i in range(N):
        diff = pos[i] - pos

        #attract/repel
        strength = -alpha * (S[i] - beta)
        F[i] = np.sum(strength[:, None] * diff, axis=0)
    return F