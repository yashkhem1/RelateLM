import os
import requests
from bs4 import BeautifulSoup

#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------------Downloading----------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
url_dict = {}
url_dict['Hindi-Hindi'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Sanskrit-Hindi'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%B8%E0%A4%82%E0%A4%B8%E0%A5%8D%E0%A4%95%E0%A5%83%E0%A4%A4-%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Pali-Hindi'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%AA%E0%A4%BE%E0%A4%B2%E0%A4%BF-%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Bengali-Hindi'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%AC%E0%A4%BE%E0%A4%82%E0%A4%97%E0%A5%8D%E0%A4%B2%E0%A4%BE-%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Marathi-Hindi'] = 'https://hi.wiktionary.org/wiki/%E0%A4%AE%E0%A4%B0%E0%A4%BE%E0%A4%A0%E0%A5%80-%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Gujarati-Hindi'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%97%E0%A5%81%E0%A4%9C%E0%A4%B0%E0%A4%BE%E0%A4%A4%E0%A5%80_%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Hindi-Telugu'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80-%E0%A4%A4%E0%A5%87%E0%A4%B2%E0%A5%81%E0%A4%97%E0%A5%81_%E0%A4%B5%E0%A5%8D%E0%A4%AF%E0%A4%BE%E0%A4%B5%E0%A4%B9%E0%A4%BE%E0%A4%B0%E0%A4%BF%E0%A4%95_%E0%A4%B2%E0%A4%98%E0%A5%81_%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Tamil-Telugu'] =  'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%A4%E0%A4%AE%E0%A4%BF%E0%A4%B2-%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Hindi-Tamil'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80-%E0%A4%A4%E0%A4%AE%E0%A4%BF%E0%A4%B2_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Hindi-Malayalam'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80-%E0%A4%AE%E0%A4%B2%E0%A4%AF%E0%A4%BE%E0%A4%B2%E0%A4%AE_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6-%E0%A5%A6%E0%A5%A8'
url_dict['Malayalam-Hindi'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%AE%E0%A4%B2%E0%A4%AF%E0%A4%BE%E0%A4%B2%E0%A4%AE-%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Punjabi-Hindi'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%AA%E0%A4%82%E0%A4%9C%E0%A4%BE%E0%A4%AC%E0%A5%80-%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'
url_dict['Hindi-Kashmiri'] = 'https://hi.wiktionary.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%A8%E0%A4%B0%E0%A5%80:%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80-%E0%A4%95%E0%A4%B6%E0%A5%8D%E0%A4%AE%E0%A5%80%E0%A4%B0%E0%A5%80_%E0%A4%B6%E0%A4%AC%E0%A5%8D%E0%A4%A6%E0%A4%95%E0%A5%8B%E0%A4%B6'


def download_pan_hin():
    url = url_dict['Punjabi-Hindi']
    print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.content,'html.parser')
    links = soup.select('div.mw-parser-output ol li')
    lines = []
    for link in links:
        url_ = 'https://hi.wiktionary.org/'+link.a['href']
        print(url_)
        page_ = requests.get(url_)
        soup_ = BeautifulSoup(page_.content,'html.parser')
        data = soup_.select('div.mw-parser-output ol li')
        for datum in data:
            lines.append(datum.text+'\n')
    with open('Punjabi-Hindi.txt','w') as f:
        f.writelines(lines)

def download_guj_hin():
    url = url_dict['Gujarati-Hindi']
    print(url)
    lines = []
    page_ = requests.get(url)
    soup_ = BeautifulSoup(page_.content,'html.parser')
    data = soup_.select('div.mw-parser-output ol li')
    for datum in data:
        lines.append(datum.text+'\n')
    with open('Gujarati-Hindi.txt','w') as f:
        f.writelines(lines)

def download_ben_hin():
    url = url_dict['Bengali-Hindi']
    print(url)
    lines = []
    page_ = requests.get(url)
    soup_ = BeautifulSoup(page_.content,'html.parser')
    data = soup_.select('div.mw-parser-output p')
    for datum in data:
        lines.append(datum.text)
    with open('Bengali-Hindi.txt','w') as f:
        f.writelines(lines)

#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------------Parsing----------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

def parse_pan_hin():
    with open('Punjabi-Hindi.txt','r') as f:
        lines = f.readlines()
    stop_words = [str(x)+"." for x in range(5)]
    with open('wiktionary.txt','w') as w:
        for line in lines:
            line = line.strip()
            l1_word = line.split("–")[0].strip().replace(" ","_")
            l2_words = [x.strip() for x in line.split(",")[1].split("/")]
            for stop_word in stop_words:
                l2_words  =[x.replace(stop_word,"").strip() for x in l2_words]
            l2_words = [x.replace(" ","_") for x in l2_words]
            w.write(l1_word+': '+" ".join(l2_words)+"\n")

def parse_guj_hin():
    with open('Gujarati-Hindi.txt','r') as f:
        lines = f.readlines()
    with open('wiktionary.txt','w') as w:
        for line in lines:
            line = line.strip()
            l1_word = line.split('--')[0].strip().replace(" ","_")
            l2_word = line.split('--')[1].split('//')[0].split(',')[0].strip().replace(" ","_")
            w.write(l1_word+": "+l2_word+"\n")

def parse_ben_hin():
    with open('Bengali-Hindi.txt','r') as f:
        lines = f.readlines()
    stop_words = ["१.","२.","$"]
    with open('wiktionary.txt','w') as w:
        for line in lines:
            l1_word = line.split(",")[0].strip()
            try:
                l2_words = line.split(",")[2].strip().split("/")
            except IndexError:
                print(line)
                continue
            for s in stop_words:
                l2_words = [x.replace(s,"").strip() for x in l2_words]
            l2_words = [x[:x.find("{")].strip() for x in l2_words]
            l2_words = [x for x in l2_words if x]
            if len(l2_words):
                w.write(l1_word+': '+" ".join(l2_words)+"\n")



# if __name__=="__main__":
#     # download_pan_hin()
#     # parse_pan_hin()
#     # download_guj_hin()
#     # parse_guj_hin()
#     # download_ben_hin()
#     # parse_ben_hin()