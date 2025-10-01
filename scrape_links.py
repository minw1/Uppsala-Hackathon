from urllib.request import urlopen
from bs4 import BeautifulSoup
import json
import os

json_file = "links.json"

# URL of the webpage
url = "https://www.uppsala.se/kommun-och-politik/organisation/forvaltningar/arbetsmarknadsforvaltningen/information-och-tjanster/"

# Open the URL
response = urlopen(url)

# Parse the HTML
soup = BeautifulSoup(response, "html.parser")

# Extract all links
links = [a['href'] for a in soup.find_all('a', href=True)]
links = [a for a in links if "https:" in a]
for link in links:
    print(link)

confirm = input("Add to links? Y/N\n")

if confirm == "Y":
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # --- Step 3: Append new links ---
    data.extend(links)
    data = list(set(data))


    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
else:
    print("aborted")