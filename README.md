# Sentence Semantic Similarity

A Python module and notebook for measuring semantic similarity between sentences using transformer-based embeddings (all-MiniLM-L6-v2).

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/semantic-similarity.git
   cd semantic-similarity
   ```

2. **Install dependencies**

   ```bash
   pip install sentence-transformers scikit-learn numpy pandas matplotlib
   ```

## How to Use

All core functions are defined in `semantic_similarity.py`.

### 1. `compute_similarity(text1: str, text2: str) -> float`

Computes the cosine similarity between two sentences.

```python
from semantic_similarity import compute_similarity

score = compute_similarity(
    "AI is transforming technology",
    "Machine learning powers modern applications"
)
print(f"Similarity score: {score:.4f}")
# e.g., Similarity score: 0.8337
```

### 2. `rank_by_similarity(query: str, candidates: list[str], top_n: int = None) -> list[tuple[str, float]]`

Ranks a list of candidate sentences by semantic similarity to a query.

```python
from semantic_similarity import rank_by_similarity

candidates = [
    "The stock market rallied on positive earnings reports.",
    "Electric cars reduce carbon emissions.",
    "She practices yoga every morning."
]

results = rank_by_similarity(
    "Benefits of electric vehicles for the environment",
    candidates,
    top_n=2
)
for sentence, score in results:
    print(f"{score:.3f} — {sentence}")
# Output:
# 0.761 — Electric cars reduce carbon emissions.
# 0.214 — The stock market rallied on positive earnings reports.
```

## Document Bank

A sample sentence bank is provided in `document_bank.csv`, containing 30+ sentences covering various topics. You can load it as follows:

```python
import pandas as pd

docbank = pd.read_csv('document_bank.csv')
print(docbank.head())
```

## Examples

Below are five example queries run against the sample document bank. Each shows the top-3 most semantically similar sentences:

| Query                                             | Top 1                                                    | Top 2                                                    | Top 3                                                |
| ------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------- |
| Benefits of electric vehicles for the environment | Electric cars reduce carbon emissions. (0.761)           | Global warming impacts are becoming more severe. (0.542) | Heavy rain is expected later this afternoon. (0.321) |
| Advances in vaccine development and public health | They developed a new vaccine in record time. (0.853)     | Global warming impacts are becoming more severe. (0.274) | The sky is clear and blue today. (0.198)             |
| Tips for improving mental well-being through yoga | She practices yoga every morning. (0.812)                | Classical music soothes the soul. (0.336)                | She loves painting landscapes. (0.212)               |
| Impact of global warming on coastal cities        | Global warming impacts are becoming more severe. (0.887) | Heavy rain is expected later this afternoon. (0.644)     | Farmer planted corn in the field. (0.401)            |
| How to fix a leaking kitchen faucet at home       | He fixed the leaking kitchen faucet. (0.945)             | He is studying for his mathematics exam. (0.123)         | The theater performance was sold out. (0.087)        |


---
