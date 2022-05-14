

from numpy.lib.arraysetops import isin
import spacy
import ginza

import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

from tqdm import tqdm

nlp = spacy.load('ja_ginza')
import collections

from tqdm import tqdm

import MeCab
from wakame.tokenizer import Tokenizer
from wakame.analyzer import Analyzer
from wakame.charfilter import *
from wakame.tokenfilter import *
tokenizer_ = Tokenizer(use_neologd=True)

# ginza のプリセット
pos_preset = [
    "pad",
    "FOS",
    "EOS",
    
    "名詞-普通名詞-一般",
    "名詞-普通名詞-サ変可能" ,
    "名詞-普通名詞-形状詞可能" ,
    "名詞-普通名詞-サ変形状詞可能" ,
    "名詞-普通名詞-副詞可能",
    "名詞-普通名詞-助数詞可能",
    "名詞-固有名詞-一般",
    "名詞-固有名詞-人名-一般",
    "名詞-固有名詞-人名-姓",
    "名詞-固有名詞-人名-名",
    "名詞-固有名詞-地名-一般",
    "名詞-固有名詞-地名-国",
    "名詞-数詞",
    "名詞-助動詞語幹",
    "代名詞",

    "形状詞-一般",
    "形状詞-タリ",
    "形状詞-助動詞語幹",

    "連体詞",
    "副詞",
    "接続詞",

    "感動詞-一般" ,
    "感動詞-フィラー" ,

    "動詞-一般" ,
    "動詞-非自立可能",

    "形容詞-一般",
    "形容詞-非自立可能",

    "助動詞",

    "助詞-格助詞",
    "助詞-副助詞",
    "助詞-係助詞",
    "助詞-接続助詞",
    "助詞-終助詞",
    "助詞-準体助詞",

    "接頭辞",
    "接尾辞-名詞的-一般",
    "接尾辞-名詞的-サ変可能",
    "接尾辞-名詞的-形状詞可能",
    "接尾辞-名詞的-サ変形状詞可能",
    "接尾辞-名詞的-副詞可能",
    "接尾辞-名詞的-助数詞",
    "接尾辞-形状詞的",
    "接尾辞-動詞的",
    "接尾辞-形容詞的",

    "記号-一般",
    "記号-文字",

    "補助記号-一般",
    "補助記号-句点",
    "補助記号-読点",
    "補助記号-括弧開",
    "補助記号-括弧閉",
    "補助記号-ＡＡ-一般",
    "補助記号-ＡＡ-顔文字",
    "空白",
]

filler_func = lambda L: ["FOS", "FOS",  *L, "EOS", "EOS"]
filler_func_one = lambda L: ["FOS",  *L, "EOS"]
# filler_func_google = lambda L: ["FOS",  *L, "EOS"]
filler_func_sep = lambda L: [*L, "[SEP]"]


independent_set = set("NOUN PROPN VERB ADJ ADV PRON NUM".split())
toyoshima_set = set("NOUN PROPN VERB ADJ".split())

import neologdn
import re
from sudachipy import tokenizer
from sudachipy import dictionary
tokenizer_obj = dictionary.Dictionary().create()
tmode = tokenizer.Tokenizer.SplitMode.C

def clean_text(text):
    text_ = neologdn.normalize(text)
    text_ = re.sub(r'\([^\)]*\)', "", text_)
    text_ = re.sub(r'\d+', "0", text_)
    text_  = "".join( [m.normalized_form() if m.part_of_speech()[0]=="名詞" else m.surface() for m in tokenizer_obj.tokenize(text_, tmode)] )
    text_ = text_.replace("??", "?")
    return text_

def normalized_span(text):
    return [m.normalized_form() for m in tokenizer_obj.tokenize(text, tmode)]

def mecab_tokenize(text):
    return tokenizer_.tokenize(text, wakati=True)

def ginza_tokenize(text):
    return [t.otrh_ for t in nlp(text)]

def fill_SYMBOL(L):
    return list(map(filler_func, L))

def fill_SYMBOL_ONE(L):
    return list(map(filler_func_one, L))

def fill_SYMBOL_SEP(L):
    sep = list(map(filler_func_sep, L[:-1]))
    sep.append(L[-1])
    return sep

def get_all_pos_dict():
    return dict( zip(pos_preset, range(len(pos_preset))) )

# 修辞
from tqdm import tqdm
def rhetoricasl_and_words(sen):
    # docs = sentence2docs(sen, sents_span=False)
    rhetoricasl = []
    for s in tqdm( sen ) :
        doc = nlp(s)
        phrases = ginza.bunsetu_phrase_spans(doc)
        phrase_otrh = [ str(p) for p in phrases ]
        rhetoricasl.append( list ( set( phrase_otrh + [token.lemma_ for token in doc] )  ) )
        # rhetoricasl.extend( )
    return rhetoricasl
    # return rhetoricasl

def extract_independet(sen):
    docs = sentence2docs(sen, sents_span=False)
    independent = []
    for doc in docs:
        words = []
        for token in doc:
            if token.pos_ in independent_set:
                    # print(token.lemma_)
                words.append(token.lemma_)
            # else:
            #      words.append(token.orth_)
        independent.append(words)
    return independent

def sentence2docs(sen, sents_span=True):
    if isinstance(sen, str):
        doc = nlp(sen)
        # 普通の処理
        if sents_span:
            texts = [str(s)  for s in doc.sents]
        # 文章で区切らない
        else:
            texts = [sen]
    
    elif isinstance(sen, list):
        texts = []
        if sents_span:
            docs = list(nlp.pipe(sen, disable=['ner']))
            for doc in docs:
                texts.extend( [str(s) for s in doc.sents] )
        # 区切らない
        else:
            texts = sen
    else:
        return None
    
    docs = list(nlp.pipe(texts, disable=['ner']))

    return docs

def sentence2pos(sen, sents_span=True) -> list:
    pos_list = []
    docs = sentence2docs(sen, sents_span)
    for doc in docs:
        pos_list.append([ token.tag_ for token in doc ])    
    return pos_list

def sentence2morpheme(sen, sents_span=True)-> list:
    docs = sentence2docs(sen, sents_span)
    morpheme_list = []
    for doc in docs:
        morpheme = []
        for token in doc:
            morpheme.append(token.orth_)
        morpheme_list.append(morpheme)
    return morpheme_list

def sentence2normalize_nv(sen, sents_span=True) -> list:
    normalize_sen = []
    docs = sentence2docs(sen, sents_span)
    for doc in docs:
        words = []
        for token in doc:
            tag = token.tag_.split("-")[0]
                # print(tag)
            if tag in ["名詞", "動詞"]:
                    # print(token.lemma_)
                words.append(token.tag_)
            else:
                 words.append(token.orth_)
        normalize_sen.append(words)
    return normalize_sen

def sentence2normalize_noun(sen, sents_span=True) -> list:
    normalize_sen = []
    docs = sentence2docs(sen, sents_span)
    for doc in docs:
        words = []
        for token in doc:
            tag = token.tag_.split("-")[0]
                # print(tag)
            if tag in ["名詞"]:
                    # print(token.lemma_)
                words.append(token.tag_)
            else:
                 words.append(token.orth_)
        normalize_sen.append(words)
    return normalize_sen

def sentence2normalize_independent(sen, sents_span=True) -> list:
    normalize_sen = []
    docs = sentence2docs(sen, sents_span)
    for doc in docs:
        words = []
        for token in doc:
            # tag = token.tag_.split("-")[0]
                # print(tag)
            if token.pos_ in independent_set:
                    # print(token.lemma_)
                words.append(token.tag_)
            else:
                 words.append(token.orth_)
        normalize_sen.append(words)
    return normalize_sen              

def is_contain_independent(text:str) -> bool:
    doc = nlp(text)
    for token in doc:
        for token in doc:
            if token.pos_ in independent_set:
                return True
    
    return False


def score(test, pred):
    if len(collections.Counter(pred)) <= 2:
        print('confusion matrix = \n', confusion_matrix(y_true=test, y_pred=pred))
        print('accuracy = ', accuracy_score(y_true=test, y_pred=pred))
        print('precision = ', precision_score(y_true=test, y_pred=pred))
        print('recall = ', recall_score(y_true=test, y_pred=pred))
        print('f1 score = ', f1_score(y_true=test, y_pred=pred))
    else:
        print('confusion matrix = \n', confusion_matrix(y_true=test, y_pred=pred))
        print('accuracy = ', accuracy_score(y_true=test, y_pred=pred))

