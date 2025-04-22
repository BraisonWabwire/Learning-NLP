from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample corpus (a collection of documents)
corpus = [
    "The cat sat on the mat",
    "The dog barked at the cat",
    "The cat meowed"
]

vectorizer=CountVectorizer()

X=vectorizer.fit_transform(corpus)
# print(x)

# Get feature names (i.e., vocabulary)
vocabulary = vectorizer.get_feature_names_out()

# Convert the BoW matrix to an array
bow_array = X.toarray()

# Print results
print("Vocabulary:\n", vocabulary)
print("\nBag of Words Matrix:")
for i, row in enumerate(bow_array):
    print(f"Document {i+1}:", row)

