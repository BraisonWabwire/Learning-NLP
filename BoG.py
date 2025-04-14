from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus (a collection of documents)
corpus = [
    "The cat sat on the mat",
    "The dog barked at the cat",
    "The cat meowed"
]

vectorizer=CountVectorizer()

x=vectorizer.fit_transform(corpus)
