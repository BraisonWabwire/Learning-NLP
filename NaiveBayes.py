# Multinomial naive bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = ["I love programming", "Spam message", "Machine learning is amazing", "Win money now"]
labels = ["ham", "spam", "ham", "spam"]

model=make_pipeline(CountVectorizer(),MultinomialNB())
model.fit(texts,labels)

print(model.predict(["Win a price"])) 