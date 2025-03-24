from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

text="My name is Braison this is my first NLTk programming example, i enjoy it"

words=word_tokenize(text)
# Removing stopwords
text_free_tokens=[word for word in words if word.lower() not in stopwords.words('english')]
print(text_free_tokens)


text_tag=pos_tag(text_free_tokens)
print(text_tag)