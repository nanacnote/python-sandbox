import os
import requests
import lxml
from bs4 import BeautifulSoup


base = "https://marvel.fandom.com"
# this is a list of the avengers in Earth-616
url = "https://marvel.fandom.com/wiki/Category:Avengers_(Earth-616)/Members"

f = requests.get(url)

soup = BeautifulSoup(f.content, "lxml")

# scan the avengers member list to find links to each of there pages.
characters = soup.find(
    'div', {"class": "category-page__members"}).find_all('a')

# make a generator of unique urls to run through.
new_urls = (character['href'] for character in characters)

pages = ((url, requests.get(base+url)) for url in new_urls)

os.makedirs(os.path.join(os.getcwd(), "wiki"), exist_ok=True)

for url, page in pages:
    f = open(f'.{url}.html', 'wb')
    f.write(page.content)
    f.close()
