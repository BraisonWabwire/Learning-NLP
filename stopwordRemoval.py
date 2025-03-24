from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text="My name is Braison this is my first NLTk programming example, i enjoy it"

words=word_tokenize(text)
# Removing stopwords
text_free_tokens=[word for word in words if word.lower() not in stopwords.words('english')]
print(text_free_tokens)




