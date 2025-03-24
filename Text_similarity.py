# Cosine similariy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text=["I love NLP","NLP is amaizing"]
vecorizer=TfidfVectorizer()

vectors=vecorizer.fit_transform(text)
print(vectors)

similarity=cosine_similarity(vectors[0],vectors[1])
print(similarity)