from nltk.util import ngrams
from nltk.tokenize import word_tokenize

text="My name is Braison this is my first NLTk programming example, i enjoy it"

tokens=word_tokenize(text)

bigrams=list(ngrams(tokens,2))
trigrams=list(ngrams(tokens,3))
print(bigrams)
print(trigrams)