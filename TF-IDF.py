from sklearn.feature_extraction.text import TfidfVectorizer

documents=["My name is Braison", "Braison was the name given by my parents", "My parents love me as Braison Wabwire"]

tf_idf=TfidfVectorizer()
X=tf_idf.fit_transform(documents)
print(X.toarray())