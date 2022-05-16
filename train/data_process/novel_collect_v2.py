import requests
import yaml
import json
import gzip
import pandas as pd

from urllib import request
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm


def get_one_page(url):
    # url = "http://www.yomiuri.co.jp/"
    text_list = []
    try:
        html = request.urlopen(url)

        #set BueatifulSoup
        soup = BeautifulSoup(html, "html.parser")

        #get 本文
        contents = soup.find("div", id="novel_honbun", class_="novel_view")
        contents_p = contents.find_all("p")
        
        for text in contents_p:
            # line = re.sub("<.+>", "", str(line))
            text = text.get_text().strip()
            if not text :
                continue
            text_list.append(text)

        return text_list
        
    except Exception as E:
        return text_list

base_url = "https://ncode.syosetu.com/"
def crawl_one_novel_div_page(ncode:str, start=1, end=10):
    novel_url = base_url+ncode.lower()+"/"
    novel = []
    for i in range(start, end+1) :
        url = novel_url+str(i)
        page = get_one_page(url)
        if page==[]:
            break
        novel.append(page)
    return novel


class Collector2:
    def __init__(self, out_path="../../corpus/novel2/"):
        self.out_path = out_path
        self.utt_set = set()
        self.title_set = set()
    
    def crawl(self, limit=5000, num_page=100):
        # limit 以下の場合に実行
        
        # 小説タイトルが沢山
        ncodes = self.get_titles(limit)
        for i, ncode in enumerate(tqdm(ncodes)):
            
            # タイトルが既に登録されていれば無視
            if ncode in self.title_set:
                continue
            # print(ncode)
            # 会話文のみ
            contents = self.extract_contents_from_ncode(ncode, end=num_page)
            # self.utt_set.update(set(convs))
            # 小説の保存
            self.write(ncode, contents)
                # 追加したタイトルを保存
            self.title_set.add(ncode)

            # 進捗
            if (i+1) % 10 == 0:
                print("complete rate => {0} : {1}%".format(i+1, (i+1)*100//len(ncodes)))
        
        print("ended crawl/ crawled data => {0} : {1}%".format(i+1, (i+1)*100//len(ncodes)))
            
    
    def extract_contents_from_ncode(self, ncode, end=100) -> list:
        novel = crawl_one_novel_div_page(ncode, end=end)
        return novel
    
    def get_titles(self, limit):
        payload = {'out': 'json','gzip':5,'order':'yearlypoint','lim':limit}
        res = requests.get("https://api.syosetu.com/novelapi/api/?genre=102&length=100000-&type=r", params=payload, timeout=30).content
        r =  gzip.decompress(res).decode("utf-8")
        df_temp = pd.read_json(r)
        df_temp = df_temp.drop(0)
        ncodes = list(df_temp.ncode.values)
        return ncodes
    
    def save(self, filename, titles):
        # conv_data = dict()
        # conv_data["utt"] = sorted( list(self.utt_set) ) 

        title_data = dict()
        title_data["ncode"] = sorted( list(self.title_set) ) 

        # with open(self.out_path+filename, mode="w") as f:
        #     json.dump(conv_data, f, ensure_ascii=False, indent=4)
        
        with open(self.out_path+titles, mode="w") as f:
            json.dump(title_data, f, ensure_ascii=False, indent=4)
        
    def load(self, filename, titles):
        # 実データ
        # if not os.path.exists(self.out_path+filename):
        #     print("file was not found : {0}".format(self.out_path+filename))
        #     return False
        # else:
        #     # jsonファイル
        #     with open(self.out_path+filename, mode="r") as f:
        #         data = json.load(f)
        #     self.utt_set = set(data["utt"])
        #     print("success load : {0}".format(self.out_path+filename))
        
        # 読み込んだタイトル達
        if not os.path.exists(self.out_path+titles):
            print("file was not found : {0}".format(self.out_path+titles))
            return False

        else:
            with open(self.out_path+titles, mode="r") as f:
                data = json.load(f)
            self.title_set = set(data["ncode"])
            print("success load : {0}".format(self.out_path+titles))
        
        self.title_set.update(set(os.listdir(self.out_path)))
        
        return True

    def write(self, ncode, contents):
        os.makedirs(self.out_path+ncode, exist_ok=True)
        for i, content in enumerate(contents):
            # print(content)
            with open(self.out_path+ncode+"/"+"{0}.txt".format(i+1), "w") as f:
                f.write("\n".join(content))


titles = "ncodes_{0}.json".format(50)
col = Collector2(out_path="../../corpus/novel/")
col.load("", titles)
col.crawl(limit=500, num_page=100)
col.save("", titles)