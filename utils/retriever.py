
import numpy as np

def retrieve_chunks(index, query_embedding, chunks, k=5):

    if index is None or not chunks:
        return []

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    k = min(k, len(chunks))

    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []

    for i in indices[0]:
        if i != -1 and i < len(chunks):
            retrieved_chunks.append(chunks[i])

    return retrieved_chunks
