def save_embeddings(embeddings, filename):
    with open(filename, 'w') as f:
        for embedding in embeddings:
            f.write(','.join(map(str, embedding)) + '\n')

def load_embeddings(filename):
    embeddings = []
    with open(filename, 'r') as f:
        for line in f:
            embeddings.append(list(map(float, line.strip().split(','))))
    return embeddings

def create_directory(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)