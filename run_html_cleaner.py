from bs4 import BeautifulSoup
import os

def run_html_cleaner(path, download=True):
    file_name = "{}_clean.txt".format(path[:-4])
    if os.path.isfile(file_name):
        document = open(file_name, 'r', encoding='utf-8')
        return document, file_name

    with open(path, mode='r', encoding='utf-8') as raw_html:
        soup = BeautifulSoup(raw_html, "lxml")

        while len(soup.find_all("table")) > 0:
            soup.table.decompose()

        while len(soup.find_all("img")) > 0:
            soup.img.decompose()
        document = ""
        for _span in soup.find_all("span"):
            if _span.text == None:
                pass
            else:
                document += str(_span.text) + "\n"

    if download:
        with open(file_name, "w", encoding='UTF-8') as processed_text:
            processed_text.write(document)
            processed_text.close()

    return processed_text, file_name