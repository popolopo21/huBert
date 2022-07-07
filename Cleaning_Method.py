from bs4 import BeautifulSoup
import re
from Abbrevations import Abbrevations
from tqdm.auto import tqdm
import glob

#This class is responsible for the text cleaning.
class CleaningMethods:
    def __init__(self, abbrev_path):
        self.characters =['a','A','á','Á','b','B','c','C','d','D','e','E','é','É','f','F','g','G','h','H','i','I','í','Í','j','J','k','K','l','L','m','M','n','N','o','O','ó','Ó','ö','Ö','ő','Ő','p','P','q','Q','r','R','s','S','t','T','u','U','ú','Ú','ü','Ü','ű','Ű','v','V','w','W','x','X','y','Y','z','Z','.','?',',','!']
        self.abbrev_path = abbrev_path

    #This method reads the given text files, cleans them and after it creates new files to the given folder
    def clean_files(self, pathsfrom, pathsto):
        
        cleaned_count = 0
        for path in tqdm(pathsfrom):
            with open(path,'r+',encoding='utf-8') as f:
                text = f.read()
                cleaned_text = self.clean_text(text)
                with open(f'{pathsto}/file_{cleaned_count}.txt', 'w', encoding='utf-8') as fp:
                    fp.write(''.join(cleaned_text))
                cleaned_count += 1

    #This method uses all cleaning method on the given text
    def clean_text(self,text):     ## sorrendre figyelni!!!!!!
        cleaned = ""
        cleaned = self.remove_htmls(text)
        self.remove_urls(cleaned)
        cleaned = self.remove_mentions(cleaned)
        cleaned = self.remove_wordswithnumbers(cleaned)
        self.extract_abbrevations(cleaned, self.abbrev_path)
        cleaned_words = self.split(cleaned)
        self.remove_words(cleaned_words)
        self.remove_special(cleaned_words)
        
        return self.join(cleaned_words)

    #This method removes every word with more than 44 characters
    def remove_words(words):
        for word in words:
            if(len(word)>44):
                words.remove(word)

    #This method removes every word which contains not valid character.
    def remove_special(self,words):
        for word in words:
            i=0
            good = True
            while(i<len(word) and good):
                if(self.characters.__contains__(word[i])):
                    i+=1
                else:
                    good = False
            if(not good):
                word = ""

    #This method removes html tags
    def remove_htmls(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    #This method removes every urls.
    def remove_urls(text):
        text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)

    #This method extracts every abbrevations
    def extract_abbrevations(text, path):
        abbr = Abbrevations()
        abbr.load_abbrevations(path)
        abbr.extract_abbrevations(text)

    #This method splits strings into list of words.
    def split(text):
        return text.split()

    #This method creates string from a list.
    def join(text):
        return " ".join(text)