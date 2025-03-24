import spacy

text="My name is Braison this is my first NLTk programming example, i enjoy it"

nlp = spacy.load("en_core_web_sm")
doc=nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)