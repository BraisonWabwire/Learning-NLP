# Word2Vec
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

sentences=['I love NLP', 'NLP is amaizing']

# Tokenize each sentence separately
sent_tokens = [word_tokenize(sentence) for sentence in sentences]

# Print tokenized sentences
print(sent_tokens)

model=Word2Vec(sent_tokens,vector_size=50,window=5,min_count=1,workers=4)

print(model.wv["NLP"]) 