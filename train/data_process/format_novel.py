import os
import sys
sys.path.append("../")
from datatools.analyzer import *


novel_path = "../../corpus/novel2/"
ncodes= [x  for x in os.listdir(novel_path) if "." not in x ]

def format_EOS(sentence):
    if sentence[-1] in ["。", "?", "!", "．"]:
        return sentence
    else:
         return sentence+"。"

def is_valid_utterance_segment(text:str)->bool:
    # 複数の"「"には囲まれていない
    if text.count("「") == 1 and text.count("」") == 1 and text[0]=="「" and text[-1]=="」":
        return True
    return False

def clean_text_plain(text):
    text_ = neologdn.normalize(text)
    # text_ = re.sub(r'\([^\)]*\)', "", text_)
    # text_ = re.sub(r'\([^\)]*\)', "", text_)
    text_ = re.sub(r'\d+', "0", text_)
    # if "……" in text_:
    #     text_ = text_.replace("……", "…")
    text_ = text_.replace("…", "")
    return text_

# 1ページの小説をデータに変換
import numpy as np
import copy
pattern = '(?<=\「).*(?=\」)'
def novel2data(novel:list):
    data_list = []
    """
    1 : 会話
    0 : それ以外

    011, 101, 111 の時に登録可能とする
    
    """
    # 1, 0 の登録
    # zero_one_vector = np.zeros(len(novel))
    zero_one = ""
    novel_normalized = []

    conv_data = []

    for i, line in enumerate(novel):
        # 普通の発話なら1
        if is_valid_utterance_segment(line):
            zero_one += "1"
            utt = re.search(pattern, line).group()
            # そもそも存在しないのであれば
            if utt=="":
                # print(line)
                continue
            utt = clean_text_plain(utt)
            if utt=="":
                utt = re.search(pattern, line).group()
            utt = format_EOS( utt )
            novel_normalized.append(utt)
        else:
            zero_one += "0"
            novel_normalized.append(clean_text_plain(line))

    descriptive = []
    # 部分を検索
    for i in range(2, len(novel_normalized)):
        if zero_one[i]=="0":
            descriptive.append(novel_normalized[i])
        comp = zero_one[i-3:i]
        if "011" == comp:
            conv_data.append(descriptive[-3:] + novel_normalized[i-2:i])
            
        elif "101" == comp:
            conv_data.append(descriptive[-3:] +[novel_normalized[i-3], novel_normalized[i-1]])

        elif "111" == comp:
            conv_data.append(descriptive[-3:] + novel_normalized[i-2:i])

        else:
            pass
    
    return conv_data


conv_data = []
from tqdm import tqdm
for ncode in tqdm(ncodes):
    for number in sorted(os.listdir(novel_path+ncode) ,reverse=True):
        with open(novel_path+ncode+"/"+number, "r") as f:
            novel = f.read().splitlines()
            conv_data += novel2data(novel) 

out_path = "../../corpus/novel_formated/"

corpus_name = "novel_segments2.tsv"
import csv

with open(out_path+corpus_name, "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(conv_data)