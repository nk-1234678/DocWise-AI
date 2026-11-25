
def generate_embeddings(client, texts):

    if not texts:
        raise ValueError("Text list is empty")

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    if not response.embeddings:
        raise ValueError("Failed to generate document embeddings")

    return [
        embedding.values
        for embedding in response.embeddings
    ]


def generate_query_embedding(client, query):

    if not query or not query.strip():
        raise ValueError("Query is empty")

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )

    if not response.embeddings:
        raise ValueError("Failed to generate query embedding")

    return response.embeddings[0].values
