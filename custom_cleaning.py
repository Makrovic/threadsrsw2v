import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()


class CustomCleaning:
    def __init__(self):
        self.id_stopword_dict = pd.read_csv("stopwordbahasa.csv", header=None)
        self.id_stopword_dict = self.id_stopword_dict.rename(columns={0: "stopword"})

    def clean(self, text):
        text = text.lower()  # lowercase
        text = re.sub(
            "((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))", " ", text
        )  # remove urls
        text = re.sub("\n", " ", text)  # remove \n
        text = re.sub("  +", " ", text)  # remove extra spaces
        text = re.sub("[^0-9a-zA-Z]+", " ", text)  # remove non alphanumeric
        return text

    def remove_stopword(self, text):
        text = " ".join(
            [
                "" if word in self.id_stopword_dict.stopword.values else word
                for word in text.split(" ")
            ]
        )
        text = re.sub("  +", " ", text)  # remove extra spaces
        text = text.strip()
        return text

    def stem(self, text):
        return stemmer.stem(text)
