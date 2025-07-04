import random
from itertools import combinations

# 1. 서브워드 후보 만들기
origin = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
target = "9876543210ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqponmlkjihgfedcba"

origin_data = [''.join(combi) for combi in combinations(origin, 2)]
target_data = [''.join(combi) for combi in combinations(target, 2)]

# 2. 랜덤하게 subword 생성
origin_vocab = set()
while len(origin_vocab) < 95:
    num_parts = random.randint(1, 5)
    parts = random.choices(origin_data, k=num_parts)
    new_token = ''.join(parts)
    origin_vocab.add(new_token)
origin_vocab = sorted(list(origin_vocab)) # 매번 동일한 데이터셋 생성을 위해 정렬

# origin 문자와 target 문자 간의 1:1 매핑 생성
char_map = {origin[i]: target[i] for i in range(len(origin))}

# char_map을 이용해 origin_vocab을 target_vocab으로 변환
target_vocab = [''.join(char_map[char] for char in token) for token in origin_vocab]

# 3. 토큰 ↔ 인덱스 매핑 생성
origin_to_index = { "<unk>" : 0, "<pad>" : 1, "<sos>" : 2, "<eos>" : 3 }
origin_to_index.update({token: idx+4 for idx, token in enumerate(origin_vocab)})
index_to_origin = {idx: token for token, idx in origin_to_index.items()}

target_to_index = { "<unk>" : 0, "<pad>" : 1, "<sos>" : 2, "<eos>" : 3 }
target_to_index.update({token: idx+4 for idx, token in enumerate(target_vocab)})
index_to_target = {idx: token for token, idx in target_to_index.items()}

# 랜덤 문장 생성: 각 문장은 subword token들의 리스트
def generate_random_sentence(subwords, min_len=10, max_len=35):
    length = random.randint(min_len, max_len)
    return random.choices(subwords, k=length)

# 문장을 토큰 ID로 변환
def tokenize(sentence_tokens, token_to_index):
    return [token_to_index[token] for token in sentence_tokens]

# 토큰 ID를 다시 서브워드로, 그리고 원문 reconstruct
def detokenize(token_ids, index_to_token):
    return [index_to_token[idx] for idx in token_ids]

# 문장 1000개 랜덤 생성
origin_sentences = [generate_random_sentence(origin_vocab) for _ in range(1000)]

# 토큰화 후 역변환하여 target_senences 생성
tokenized_origin_sentences = [tokenize(sentence, origin_to_index) for sentence in origin_sentences]
target_sentences = [detokenize(token_ids, index_to_target) for token_ids in tokenized_origin_sentences]

import json

# 디렉토리 또는 파일명은 자유롭게 수정 가능
with open("origin_to_index.json", "w", encoding="utf-8") as f:
    json.dump(origin_to_index, f, ensure_ascii=False, indent=2)

with open("target_to_index.json", "w", encoding="utf-8") as f:
    json.dump(target_to_index, f, ensure_ascii=False, indent=2)

# origin_sentences를 텍스트로 저장
with open("origin_sentences.txt", "w", encoding="utf-8") as f:
    for sentence in origin_sentences:
        f.write(" ".join(sentence) + "\n")

# target_sentences도 동일하게 저장
with open("target_sentences.txt", "w", encoding="utf-8") as f:
    for sentence in target_sentences:
        f.write(" ".join(sentence) + "\n")