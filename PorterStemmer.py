from nltk. stem import PorterStemmer

words=input('Enter a words comma separated:'+" ")
words=[words]
stemmer=PorterStemmer()
stem=[stemmer.stem(word) for word in words ]
print(stem)
