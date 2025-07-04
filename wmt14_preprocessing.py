# !python -m spacy download en
# !python -m spacy download de

import pandas as pd

data = pd.read_csv("wmt14_translate_de-en_test.csv")

import spacy

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

# 독일어(Deutsch) 문장을 토큰화 하는 함수
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]
de_token = data['de'].apply(tokenize_de)

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]
en_token = data['en'].apply(tokenize_en)

from collections import defaultdict

# 입력 최대 토큰 값을 15000으로 제한
MAX_LEN = 15000

# 독일어(Deutsch) 토큰을 숫자 인덱스로 변환
de_dict = defaultdict(list)
de_dict["<unk>"] = 0
de_dict["<pad>"] = 1
de_dict["<sos>"] = 2
de_dict["<eos>"] = 3
de_vocab, idx = [], 4
for token_list in de_token:
    tmp = [2] # 시작 토큰 추가
    for token in token_list:
        if token not in de_dict:
            de_dict[token] = idx
            idx += 1
        tmp.append(de_dict[token])
    tmp.append(3) # 종료 토큰 추가
    # 패딩 처리
    if len(tmp) < MAX_LEN:
        tmp += [1] * (MAX_LEN - len(tmp))
    de_vocab.append(tmp)
print("독일어 토큰 사전 크기:", len(de_dict))

# 영어(English) 토큰을 숫자 인덱스로 변환
en_dict = defaultdict(list)
en_dict["<unk>"] = 0
en_dict["<pad>"] = 1
en_dict["<sos>"] = 2
en_dict["<eos>"] = 3
en_vocab, idx = [], 4
for token_list in en_token:
    tmp = [2] # 시작 토큰 추가
    for token in token_list:
        if token not in en_dict:
            en_dict[token] = idx
            idx += 1
        tmp.append(en_dict[token])
    tmp.append(3) # 종료 토큰 추가
    # 패딩 처리
    if len(tmp) < MAX_LEN:
        tmp += [1] * (MAX_LEN - len(tmp))
    en_vocab.append(tmp)
print("영어 토큰 사전 크기:", len(en_dict))