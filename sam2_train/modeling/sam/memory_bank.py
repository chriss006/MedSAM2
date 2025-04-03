import torch
import torch.nn.functional as F

class SelfSortingMemoryBank:
    def __init__(self, max_size=16, similarity_fn=None, conf_thresh=0.5):
        self.max_size = max_size
        self.memory = []  # List[Tensor]
        self.conf_thresh = conf_thresh
        self.similarity_fn = similarity_fn or (lambda x, y: F.cosine_similarity(x.flatten(), y.flatten(), dim=0))

    def update(self, embedding: torch.Tensor, confidence: float):
        if confidence < self.conf_thresh:
            return

        # Add current embedding to candidate memory
        candidate_memory = self.memory + [embedding]

        # Compute dissimilarity score for each embedding
        dissimilarities = []
        for i, emb_i in enumerate(candidate_memory):
            D_i = 0
            for j, emb_j in enumerate(candidate_memory):
                if i == j:
                    continue
                sim = torch.clamp(self.similarity_fn(emb_i, emb_j), min=0.0)
                D_i += (1 - sim)
            dissimilarities.append((D_i, emb_i))

        # Keep top-K dissimilar embeddings
        sorted_memory = sorted(dissimilarities, key=lambda x: x[0], reverse=True)
        self.memory = [item[1] for item in sorted_memory[:self.max_size]]

    def resample(self, current_embedding: torch.Tensor):
        if not self.memory:
            return current_embedding  # fallback

        sims = torch.tensor([self.similarity_fn(current_embedding, m) for m in self.memory])
        probs = sims / (sims.sum() + 1e-8)  # avoid division by zero
        weighted_memory = torch.stack(self.memory, dim=0)  # [N, C]
        return torch.sum(probs[:, None] * weighted_memory, dim=0)
