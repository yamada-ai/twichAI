import os
import sys
from datatools.analyzer import *

from datatools.maneger import DataManager
import pandas as pd

import csv
import time

import random
random.seed(0)

from transformer_model import *

import copy
from collections import Counter
from torchtext.vocab import Vocab

PAD_IDX = 1 
START_IDX = 2
END_IDX = 3
device = "cpu"


def make_context_added_Src_Tgt(conv_data):

    context_src_str = []
    tgt_str = []
    prev_utt = []
    current_situation = [""]
    for conv in conv_data:
        # 状況が変化したか
        if current_situation[0] != conv[0]:
            current_situation = conv[:-2]
            # エラー対策
            if current_situation==[]:
                current_situation = [""]
            prev_utt = [conv[-2]]
        
        context_src_str.append([current_situation, copy.deepcopy(prev_utt) ])
        prev_utt.append(conv[-1])
        tgt_str.append(conv[-1])
    
    return context_src_str, tgt_str

def build_vocab(texts, tokenizer):
    
    counter = Counter()
    for text in tqdm(texts):
        counter.update(tokenizer(text))
    return Vocab(counter, specials=['<unk>', '<pad>', '<fos>', '<eos>','<sep>', '<cxt>', '<del>'])

def data_process(texts_src, texts_tgt, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt):
    
    data = []
    for (src, tgt) in zip(texts_src, texts_tgt):
        src_tensor = torch.tensor(
            convert_text_to_indexes(text=src, vocab=vocab_src, tokenizer=tokenizer_src, mode="src"),
            dtype=torch.long
        )
        tgt_tensor = torch.tensor(
            convert_text_to_indexes(text=tgt, vocab=vocab_tgt, tokenizer=tokenizer_tgt, mode="tgt"),
            dtype=torch.long
        )
        data.append((src_tensor, tgt_tensor))
        
    return data

def convert_text_to_indexes(text, vocab, tokenizer, mode="src"):
    if mode=="src":
        sit = text[0]
        segments = [vocab['<sep>']]
        for s in sit:
            segments += [vocab[token] for token in tokenizer(s.strip("\n"))] + [vocab['<sep>']]
        # 最後消す
        segments[-1] = vocab['<cxt>']
        utt = text[1]
        for u in utt:
            segments += [vocab[token] for token in tokenizer(u.strip("\n"))] + [vocab['<cxt>']]
        return segments
    # 
    elif mode=="tgt":
        return [vocab['<fos>']] + [
            vocab[token] for token in tokenizer(text.strip("\n"))
        ] + [vocab['<eos>']]
    else:
        return []

def generate_batch(data_batch):
    
    batch_src, batch_tgt = [], []
    for src, tgt in data_batch:
        batch_src.append(src)
        batch_tgt.append(tgt)
        
    # batch_src = pad_sequence(batch_src, padding_value=PAD_IDX, batch_first=True)
    # batch_tgt = pad_sequence(batch_tgt, padding_value=PAD_IDX, batch_first=True)
    batch_src = pad_sequence(batch_src, padding_value=PAD_IDX, batch_first=False)
    batch_tgt = pad_sequence(batch_tgt, padding_value=PAD_IDX, batch_first=False)
    
    return batch_src, batch_tgt

def train(model, data, optimizer, criterion, PAD_IDX):
    
    model.train()
    losses = 0
    for src, tgt in tqdm(data):
        
        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, :]

        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)

        # print(src.shape, tgt.shape)

        logits = model(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )

        optimizer.zero_grad()

        output_tgt = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), output_tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        del logits
        del loss
        
    return losses / len(data)


def evaluate(model, data, criterion, PAD_IDX):
    
    model.eval()
    losses = 0
    for src, tgt in data:
        
        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, :]

        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)

        logits = model(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )
        
        output_tgt = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), output_tgt.reshape(-1))
        losses += loss.item()
        
    return losses / len(data)


def main():
    out_path = "../corpus/novel_formated/"
    corpus_name = "novel_segments.tsv"

    conv_data = []
    with open(out_path+corpus_name, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        conv_data = [row for row in reader]

    lim = 100000
    conv_data_mini = random.sample(conv_data, lim)

    src_str, tgt_str = make_context_added_Src_Tgt(conv_data_mini)

    # 学習用のデータを分割
    src_trainval_str, src_test_str, tgt_trainval_str, tgt_test_str= train_test_split(src_str, tgt_str, test_size=0.20, random_state=5)
    print("len=> src_train_val:{0}, src_test:{1}".format(len(src_trainval_str), len(src_test_str)))

    src_train_str, src_val_str, tgt_train_str, tgt_val_str= train_test_split(src_trainval_str, tgt_trainval_str, test_size=0.10, random_state=5)

    print("len=> src_train:{0}, src_val:{1}".format(len(src_train_str), len(src_val_str)))

    src_train_str_ = src_train_str[:lim]
    tgt_train_str_ = tgt_train_str[:lim]
    src_val_str_ = src_val_str[:lim//10]
    tgt_val_str_ = tgt_val_str[:lim//10]

    # トークナイザを設定
    tokenizer_src = mecab_tokenize
    tokenizer_tgt = mecab_tokenize

    vocab_src = build_vocab(src_train_str, tokenizer=tokenizer_src)
    vocab_tgt = build_vocab(tgt_train_str, tokenizer=tokenizer_tgt)


    # 辞書を保存
    vocab_path = "../models/vocab/"
    vocab_name = "vocab_CModel_src_mini_lim={0}.pickle".format(lim)
    dictM = DataManager(vocab_path)
    dictM.save_data(vocab_name, vocab_src)
    vocab_name = "vocab_CModel_tgt_mini_lim={0}.pickle".format(lim)
    dictM.save_data(vocab_name, vocab_tgt)

    # 学習と評価データを整形
    train_data = data_process(
        texts_src=src_train_str_, texts_tgt=tgt_train_str,
        vocab_src=vocab_src, vocab_tgt=vocab_tgt,
        tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt
    )
    valid_data = data_process(
        texts_src=src_val_str_, texts_tgt=tgt_val_str_,
        vocab_src=vocab_src, vocab_tgt=vocab_tgt,
        tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt
    )


    batch_size = 64
    # グローバル変数の更新
    global PAD_IDX, START_IDX, END_IDX

    PAD_IDX = vocab_src['<pad>']
    START_IDX = vocab_src['<fos>']
    END_IDX = vocab_src['<eos>']


    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

    # グローバルのデバイスを更新
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_dir_path = Path('model')
    if not model_dir_path.exists():
        model_dir_path.mkdir(parents=True)
    
    vocab_size_src = len(vocab_src)
    vocab_size_tgt = len(vocab_tgt)
    embedding_size = 320
    nhead = 8
    dim_feedforward = 100
    num_encoder_layers = 4
    num_decoder_layers = 4
    dropout = 0.1

    model = Seq2SeqTransformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        embedding_size=embedding_size,
        vocab_size_src=vocab_size_src, vocab_size_tgt=vocab_size_tgt,
        dim_feedforward=dim_feedforward,
        dropout=dropout, nhead=nhead
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(model.parameters())

    epoch = 30
    best_loss = float('Inf')
    best_model = None
    patience = 10
    counter = 0

    for loop in range(1, epoch + 1):
        
        start_time = time.time()
        
        loss_train = train(
            model=model, data=train_iter, optimizer=optimizer,
            criterion=criterion, PAD_IDX=PAD_IDX
        )
        
        elapsed_time = time.time() - start_time
        
        loss_valid = evaluate(
            model=model, data=valid_iter, criterion=criterion, PAD_IDX=PAD_IDX
        )
        
        print('[{}/{}] train loss: {:.2f}, valid loss: {:.2f}  [{}{:.0f}s] count: {}, {}'.format(
            loop, epoch,
            loss_train, loss_valid,
            str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
            elapsed_time % 60,
            counter,
            '**' if best_loss > loss_valid else ''
        ))
        
        if best_loss > loss_valid:
            best_loss = loss_valid
            best_model = model
            counter = 0
            
        if counter > patience:
            break
        
        counter += 1

    # 保存1
    torch.save(best_model.state_dict(), model_dir_path.joinpath('cmodel.pth'))

    # 保存 保険
    model_path = "../models/transformer/"
    model_name = "CModel_lim={0}_best.pickle".format(lim)
    modelM = DataManager(model_path)
    modelM.save_data(model_name, best_model)
    model_name = "CModel_lim={0}.pickle".format(lim)
    modelM.save_data(model_name, model)

if __name__ == "__main__":
    main()