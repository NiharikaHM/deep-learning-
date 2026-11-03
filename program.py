from gensim.models import Word2Vec

corpus = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["deep", "learning"]
]

model = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=2,
    min_count=1,
    workers=1
)

print("Embedding for 'learning':")
print(model.wv["learning"])

print("\nWords similar to 'learning':")
print(model.wv.most_similar("learning"))