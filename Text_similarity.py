# Cosine similariy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text=["I love NLP","NLP is amaizing"]
vecorizer=TfidfVectorizer()

vectors=vecorizer.fit_transform(text)
print(vectors)

similarity=cosine_similarity(vectors[0],vectors[1])
print(similarity)



# Jaccard similarity
def jaccard_similarity(text1, text2):
    set1, set2 = set(text1.split()), set(text2.split())
    return len(set1 & set2) / len(set1 | set2)

print(jaccard_similarity("I love NLP", "NLP is amazing")) 