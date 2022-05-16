import os
import sys
from datatools.analyzer import *

from datatools.maneger import DataManager
import pandas as pd

import csv
import time

import random
random.seed(0)

from collections import Counter
from torchtext.vocab import Vocab

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import (
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
import torch.nn.utils.rnn as rnn


PAD_IDX = 1 
START_IDX = 2
END_IDX = 3
device = "cpu"

def make_Src_Tgt(conv_data):
    src_str = []
    tgt_str = []

    for conv in conv_data:
        src_str.append(conv[-2])
        tgt_str.append(conv[-1])
    return src_str, tgt_str

def build_vocab(texts, tokenizer):
    
    counter = Counter()
    for text in tqdm(texts):
        counter.update(tokenizer(text))
    return Vocab(counter, specials=['<unk>', '<pad>','<fos>', '<eos>', '<del>'])

# ----------
def data_process(texts_src, texts_tgt, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt):
    
    data = []
    for (src, tgt) in zip(texts_src, texts_tgt):
        src_tensor = torch.tensor(
            convert_text_to_indexes(text=src, vocab=vocab_src, tokenizer=tokenizer_src),
            dtype=torch.long
        )
        tgt_tensor = torch.tensor(
            convert_text_to_indexes(text=tgt, vocab=vocab_tgt, tokenizer=tokenizer_tgt),
            dtype=torch.long
        )
        data.append((src_tensor, tgt_tensor))
        
    return data

def convert_text_to_indexes(text, vocab, tokenizer):
    return [vocab['<fos>']] + [
        vocab[token] for token in tokenizer(text.strip("\n"))
    ] + [vocab['<eos>']]


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

# ------------------------

import math
class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
        self.embedding_size = embedding_size
        
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, embedding_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        embedding_pos = torch.zeros((maxlen, embedding_size))
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den)
        embedding_pos = embedding_pos.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.embedding_pos[: token_embedding.size(0), :])

def create_mask(src, tgt, PAD_IDX):
    
    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt, PAD_IDX)
    mask_src = torch.zeros((seq_len_src, seq_len_src), device=device).type(torch.bool)

    padding_mask_src = (src == PAD_IDX).transpose(0, 1)
    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1)
    
    return mask_src, mask_tgt, padding_mask_src, padding_mask_tgt

def generate_square_subsequent_mask(seq_len, PAD_IDX):
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == PAD_IDX, float(0.0))
    return mask


class Seq2SeqTransformer(nn.Module):
    
    def __init__(
        self, num_encoder_layers: int, num_decoder_layers: int,
        embedding_size: int, vocab_size_src: int, vocab_size_tgt: int,
        dim_feedforward:int = 512, dropout:float = 0.1, nhead:int = 8
        ):
        
        super(Seq2SeqTransformer, self).__init__()

        self.token_embedding_src = TokenEmbedding(vocab_size_src, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.token_embedding_tgt = TokenEmbedding(vocab_size_tgt, embedding_size)
        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output = nn.Linear(embedding_size, vocab_size_tgt)

    def forward(
        self, src: Tensor, tgt: Tensor,
        mask_src: Tensor, mask_tgt: Tensor,
        padding_mask_src: Tensor, padding_mask_tgt: Tensor,
        memory_key_padding_mask: Tensor
        ):
        
        embedding_src = self.positional_encoding(self.token_embedding_src(src))
        memory = self.transformer_encoder(embedding_src, mask_src, padding_mask_src)
        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(
            embedding_tgt, memory, mask_tgt, None,
            padding_mask_tgt, memory_key_padding_mask
        )
        return self.output(outs)

    def encode(self, src: Tensor, mask_src: Tensor):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt: Tensor, memory: Tensor, mask_tgt: Tensor):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)

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

    src_str, tgt_str = make_Src_Tgt(conv_data_mini)

    # 学習用のデータを分割
    src_trainval_str, src_test_str, tgt_trainval_str, tgt_test_str= train_test_split(src_str, tgt_str, test_size=0.20, random_state=5)
    print("len=> src_train_val:{0}, src_test:{1}".format(len(src_trainval_str), len(src_test_str)))

    src_train_str, src_val_str, tgt_train_str, tgt_val_str= train_test_split(src_trainval_str, tgt_trainval_str, test_size=0.10, random_state=5)

    print("len=> src_train:{0}, src_val:{1}".format(len(src_train_str), len(src_val_str)))

    # トークナイザを設定
    tokenizer_src = mecab_tokenize
    tokenizer_tgt = mecab_tokenize

    vocab_src = build_vocab(src_train_str, tokenizer=tokenizer_src)
    vocab_tgt = build_vocab(tgt_train_str, tokenizer=tokenizer_tgt)


    # 辞書を保存
    vocab_path = "../models/vocab/"
    vocab_name = "vocab_transformer_src_mini_lim={0}.pickle".format(lim)
    dictM = DataManager(vocab_path)
    dictM.save_data(vocab_name, vocab_src)
    vocab_name = "vocab_transformer_tgt_mini_lim={0}.pickle".format(lim)
    dictM.save_data(vocab_name, vocab_tgt)

    # 学習と評価データを整形
    train_data = data_process(
    texts_src=src_train_str, texts_tgt=tgt_train_str,
    vocab_src=vocab_src, vocab_tgt=vocab_tgt,
    tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt
    )
    valid_data = data_process(
        texts_src=src_val_str, texts_tgt=tgt_val_str,
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
    embedding_size = 240
    nhead = 8
    dim_feedforward = 100
    num_encoder_layers = 2
    num_decoder_layers = 2
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
    patience = 5
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
    torch.save(best_model.state_dict(), model_dir_path.joinpath('translation_transfomer2.pth'))

    # 保存 保険
    model_path = "../models/transformer/"
    model_name = "transformer_lim={0}_best.pickle".format(lim)
    modelM = DataManager(model_path)
    modelM.save_data(model_name, best_model)
    model_name = "transformer_lim={0}_.pickle".format(lim)
    modelM.save_data(model_name, model)


if __name__ == "__main__":
    main()