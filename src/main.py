import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path, max_words=5000):
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

