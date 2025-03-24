import requests
from bs4 import BeautifulSoup

# Replace with the URL you want to scrape
url = "https://citizen.digital/news/high-court-lifts-orders-restricting-nairobi-hospitals-capital-expenditure-n359794"

# Send a GET request to the webpage
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract specific elements, e.g., the page title
    title = soup.title.string
    print("Title:", title)

    # Extract all paragraph text (for context)
    paragraphs = soup.find_all("p")
    # for p in paragraphs:
        # print("Paragraph:", p.get_text())
else:
    print("Failed to retrieve the webpage. Status code:", response.status_code)



# Text preprocessing
# Tokenization
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

tokens=word_tokenize(title)
print("tokens:", tokens)


# Stopword removal
stop_free_tokens=[token for token in tokens if token.lower() not in stopwords.words('english')]
print(stop_free_tokens)

title_tags=pos_tag(stop_free_tokens)
print(title_tags)
