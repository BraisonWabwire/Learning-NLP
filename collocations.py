from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import word_tokenize

words = word_tokenize("Machine learning is a subset of artificial intelligence. Machine learning techniques are widely used.")
bigram_finder=BigramCollocationFinder.from_words(words)
my_bigram=bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 5)  # Top 5 collocations
print(my_bigram)