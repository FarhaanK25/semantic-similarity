from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(text1: str, text2: str) -> float:

    emb1 = model.encode([text1], convert_to_numpy=True)
    emb2 = model.encode([text2], convert_to_numpy=True)

    sim = cosine_similarity(emb1, emb2)[0][0]

    return float(np.clip(sim, 0.0, 1.0))


def rank_by_similarity(query: str, candidates: list, top_n: int = None) -> list:

    query_emb = model.encode([query], convert_to_numpy=True)
    cand_embs = model.encode(candidates, convert_to_numpy=True)

    sims = cosine_similarity(query_emb, cand_embs)[0]

    paired = list(zip(candidates, sims))
    paired.sort(key=lambda x: x[1], reverse=True)

    paired = [(sent, float(np.clip(score,0.0,1.0))) for sent, score in paired]
    if top_n:
        return paired[:top_n]
    return paired

sentences = [
    "The cat sat on the mat.",
    "A dog barked at the mailman.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming industry.",
    "Machine learning powers many modern applications.",
    "The sky is clear and blue today.",
    "Heavy rain is expected later this afternoon.",
    "She enjoys reading science fiction novels.",
    "He bought a new smartphone last week.",
    "The stock market rallied on positive earnings reports.",
    "This restaurant serves the best pizza in town.",
    "Our flight was delayed due to bad weather.",
    "Global warming impacts are becoming more severe.",
    "She practices yoga every morning.",
    "They are planning a trip to the mountains.",
    "The theater performance was sold out.",
    "He is studying for his mathematics exam.",
    "They developed a new vaccine in record time.",
    "Classical music soothes the soul.",
    "Farmer planted corn in the field.",
    "She loves painting landscapes.",
    "The conference begins at nine AM.",
    "He fixed the leaking kitchen faucet.",
    "They watched a documentary on wildlife.",
    "Electric cars reduce carbon emissions.",
    "The cake was decorated with fresh strawberries.",
    "He solved the complex puzzle quickly.",
    "The museum houses ancient artifacts.",
    "Space exploration expands our knowledge.",
    "Her speech inspired the audience."
]

s1 = "AI and machine learning are revolutionizing technology."
s2 = "Machine learning powers many modern applications."

score = compute_similarity(s1, s2)
print(f"Similarity between:\n  1) {s1}\n  2) {s2}\n=> {score:.4f}")

example_queries = [
    "Benefits of electric vehicles for the environment",
    "Advances in vaccine development and public health",
    "Tips for improving mental well-being through yoga",
    "Impact of global warming on coastal cities",
    "How to fix a leaking kitchen faucet at home"
]

print("\nRanking sentences based on their similarity score with each of 5 example queries : ")
for i, query in enumerate(example_queries, start=1):
    print(f"\nExample Query {i}: {query}")
    top_results = rank_by_similarity(query, sentences, top_n=5)
    for rank, (sent, score) in enumerate(top_results, start=1):
        print(f"  Top {rank}: {score:.3f} — {sent}")


subset = sentences[:10]
embs = model.encode(subset, convert_to_numpy=True)
sim_matrix = cosine_similarity(embs)

plt.figure(figsize=(8,6))
plt.imshow(sim_matrix, interpolation='nearest')
plt.colorbar(label='Cosine similarity')
plt.xticks(range(len(subset)), [f"S{i+1}" for i in range(len(subset))], rotation=45)
plt.yticks(range(len(subset)), [f"S{i+1}" for i in range(len(subset))])
plt.title('Heatmap of semantic similarity for sample sentences')
plt.tight_layout()
plt.show()
