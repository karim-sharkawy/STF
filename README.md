# Semantic Turing Field (STF)
The Semantic Turing Field, A universe of words that evolves toward meaning.
> *“A universe of words that evolves toward meaning.”*

## 🧠 What is this?

This is a side project I built because I wanted to explore a weird idea:

**What if language behaved like physics?**

The **Semantic Turing Field (STF)** treats words as particles in a 2D space. These particles move, interact, and self-organize based on their **semantic relationships**, forming clusters that represent meaning.

Then, when you input a sentence, it acts like a **gravitational disturbance** — warping the entire field and reshaping how words relate to each other in real time.

It’s not meant to be practical. It’s meant to be interesting.


## ⚙️ Core Idea

* Each word = a **particle**
* Each particle has:

  * Position ( x_i \in \mathbb{R}^2 )
  * Velocity ( v_i )
  * Mass ( m_i ) (often tied to word frequency)
* Semantic relationships = **forces**
* Sentences = **external fields / gravity waves**

Over time, the system evolves into clusters that reflect meaning — but dynamically, not statically.


## 🔤 Embeddings (Why GloVe?)

This project uses **GloVe embeddings** instead of Word2Vec.

👉 [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

**Why GloVe?**

* It captures **global co-occurrence statistics**, not just local context
* Produces more **stable semantic geometry**
* Works well when you want **meaning to behave like a field**, not just neighborhoods

Since this project depends heavily on **pairwise similarity across many words**, GloVe’s global structure is a better fit.


## 🧮 The Force Equation (Why this formula?)

The core interaction between two words ( i ) and ( j ) is:

[
F_{ij} = -\alpha \cdot (\text{cosine_similarity}(i, j) - \beta)(x_i - x_j)
]

### 🔍 Breaking it down

* **( \text{cosine_similarity}(i, j) )**
  Measures semantic similarity between words

* **( (x_i - x_j) )**
  Direction vector between particles

* **( \alpha )**
  Controls overall force strength

* **( \beta )**
  A *threshold* that determines attraction vs repulsion



## 🤔 Where does this come from?

This formula is inspired by a mix of:

### 1. **Classical Physics (Hooke / Spring Systems)**

The structure:
[
F \propto (x_i - x_j)
]

is similar to spring forces, where objects pull or push based on distance.

But instead of distance determining force strength…

👉 **semantic similarity does**



### 2. **Energy-Based Models**

You can think of the system as minimizing an implicit energy:

* Similar words → lower energy when close
* Dissimilar words → lower energy when far

The force is effectively:

[
F = -\nabla E
]

So we’re simulating a system that naturally settles into **semantic equilibrium**.



### 3. **Signed Interaction Field**

The key term:

[
(\text{similarity} - \beta)
]

creates two regimes:

| Case           | Behavior   |
| -- | - |
| similarity > β | Attraction |
| similarity < β | Repulsion  |

This is important because:

* Without repulsion → everything collapses
* Without attraction → everything drifts apart

So this term creates a **balanced, self-organizing system**



## 🌊 Sentence as a “Gravity Wave”

A sentence is embedded (e.g. average of word vectors), producing a vector:

[
s = \frac{1}{n} \sum_{k=1}^{n} w_k
]

This acts like a **potential field**:

* Words similar to the sentence → pulled inward
* Others → weakly affected

This lets you literally *see meaning reshape the space*.



## 🔄 Simulation Dynamics

* Integration: Verlet / simple Euler (depending on implementation)
* Damping: prevents chaotic divergence
* Noise: keeps the system “alive”
* Optional clustering: k-means / UMAP for visualization



## 🎨 Visualization

Each word is rendered as:

* A point (or label)
* Positioned in 2D
* Colored by cluster or semantic group

As the simulation runs:

* Clusters form
* Shift
* Break
* Re-form



## 🧪 Why does this exist?

* Because semantic spaces are usually static — this makes them **alive**
* Because embeddings are geometric, but we rarely treat them as **physical systems**
* Because it's fun to watch meaning behave like a fluid

Also:

> Because no one asked for it.



## 🚀 Future Ideas

* Word birth/death (dynamic vocabulary)
* Memory effects (semantic “scars”)
* Temporal drift (language evolution)
* Real-time interaction (user input reshaping the field)
* GPU acceleration for large vocabularies



## 📦 Tech Stack

* Python
* NumPy
* scikit-learn (PCA, cosine similarity)
* Matplotlib / PyGame / VisPy (visualization)



## 📚 Resources

* GloVe (Stanford): [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
* Word embeddings overview: [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
* Energy-based models: Yann LeCun et al.



## ⚠️ Disclaimer

This is not:

* A production NLP system
* A scientifically rigorous model of cognition

It’s an experiment in thinking about **meaning as a dynamical system**.



If you want, I can also tailor this README to match your actual repo structure (scripts/, notebooks/, etc.) or add setup instructions and usage examples.
