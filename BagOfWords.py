from sklearn.feature_extraction.text import CountVectorizer

documents=["My name is Braison", "Braison was the name given by my parents", "My parents love me as Braison Wabwire"]

vetorizer=CountVectorizer()
X=vetorizer.fit_transform(documents)
vocabulary=vetorizer.get_feature_names_out

print("vocabulary", vocabulary)
print(X.toarray())
