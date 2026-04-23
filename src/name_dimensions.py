import json
import numpy as np
from ir_engine import InvertedIndexSearchEngine, UI_DIMENSION_COUNT

engine = InvertedIndexSearchEngine("src/data/inverted_index_matrix.json")

terms = list(engine.idf.keys())
term_loadings = np.stack([engine._term_latent_vector(t) for t in terms])

out = {}
for k in range(UI_DIMENSION_COUNT):
    col = term_loadings[:, k]
    pos_idx = np.argsort(col)[::-1][:15]
    neg_idx = np.argsort(col)[:15]

    print(f"\n{'='*60}\nLS{k+1}\n{'='*60}")
    print("  POSITIVE pole:")
    for i in pos_idx: print(f"    {col[i]:+.3f}  {terms[i]}")
    print("  NEGATIVE pole:")
    for i in neg_idx: print(f"    {col[i]:+.3f}  {terms[i]}")

    out[str(k + 1)] = {
        "id": k + 1,
        "label_pos": "",  
        "label_neg": "",     
        "blurb": "",
        "top_terms_pos": [{"t": terms[i], "w": round(float(col[i]), 4)}
                          for i in pos_idx[:8]],
        "top_terms_neg": [{"t": terms[i], "w": round(float(col[i]), 4)}
                          for i in neg_idx[:8]],
    }

with open("src/data/svd_dimension_names.json", "w") as f:
    json.dump(out, f, indent=2)
