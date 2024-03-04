from torch import nn
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import optim
import numpy as np
from numpy.linalg import norm
import utils.torch_util as tu


# [Word2Vec]
# Word2Vec 은 2013 년 구글에서 공개한 임베딩 모델로 단어 간의 유사성을 측정하기 위해 분포 가설을 기반으로 개발되었습니다.
# 분포 가설은 단어 간의 동시 발생 확률 분포를 이용해 단어 간의 유사성을 측정합니다.
# 문맥상 동일한 위치에 존재하는 단어끼리는 서로 비슷한 의미를 지닌다는 것으로,
# 내일 '자동차'를 타고 부산에 간다 와 내일 '비행기'를 타고 부산에 간다 라는 두 문장에서
# 자동차와 비행기는 문맥상의 유사도와 탈것이라는 의미적 유사성을 지니고 있음을 알 수 있습니다.
# 즉, 문장에서 특정 단어가 들어가는 문맥상 특정 부위를 비워둠으로써 해당 위치에 올 단어를 예측하는 모델을 학습시키면 해당 모델은 단어의 의미를 품고 있는
# 투사층을 가지게 되는 것입니다. (학습시에는 예측 부분까지 사용하고, 의미 압축시에는 예측 부분을 제외한 투사층의 벡터가 의미 벡터 역할을 합니다.)
# 입력층에는 고유한 단어를 의미 하는 One-Hot-Vector 을 n 개 넣어주고, 이를 투사층으로 압축한 후,
# 출력층으로는 예측하는 단어의 One-Hot-Vector 를 출력하도록 학습하면, 단어의 의미를 압축하는 임베딩 모델을 만들 수 있습니다.
# 위와 같은 방법을 CBoW 라고 합니다.
# CBoW(Continuous Bag of Words) 는 위와 같이 주변 단어들을 가지고 해당 구역의 하나의 단어를 출력하도록 학습하는 방식입니다.
# 다른 방식으로는 Skip-gram 이 있으며, 이는 중심 단어 하나를 입력받아서, 주변 단어들을 예측하도록 하여 학습하는 방식입니다.
# 일반적으로는 Skip-Gram 이 CBoW 보다 성능이 좋다고 합니다.

# todo [계층적 소프트맥스]
# todo [Negative Sampling]

# 기본 형식 Skip-gram 구현
class VanillaSkipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size
        )

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        output = self.linear(embeddings)  # vocab_size 의 one-hot-vector 반환
        return output


# 네이버 영화 리뷰 데이터셋 말뭉치 로딩
corpus = Korpora.load("nsmc")

# 학습 데이터 프레임 추리기
corpus = pd.DataFrame(corpus.test)

# 한국어 형태소로 분리
tokenizer = Okt()
tokens = [tokenizer.morphs(review) for review in corpus.text]

# 형태소 분리된 문장 리스트 3개 출력 :
# [
#   ['굳', 'ㅋ'],
#   ['GDNTOPCLASSINTHECLUB'],
#   ['뭐', '야', '이', '평점', '들', '은', '....', '나쁘진', '않지만', '10', '점', '짜', '리', '는', '더', '더욱', '아니잖아']
# ]
print(tokens[:3])


# 형태소 분리된 문장 리스트들의 리스트로 단어사전 생성 함수
def build_vocab(corpus, n_vocab, special_token):
    counter = Counter()
    for tokens in corpus:
        # tokens ex : ['굳', 'ㅋ']
        counter.update(tokens)
    # 사전에 없는 단어는 모두 special_token 으로 처리할 것
    vocab = [special_token]
    for token, count in counter.most_common(n_vocab):
        vocab.append(token)
    return vocab


# tokens 으로 단어사전 생성
vocab = build_vocab(corpus=tokens, n_vocab=5000, special_token="<unk>")

# 단어사전 인덱스를 기반으로 토큰을 인덱스로 변경하는 dict
token_to_id = {token: idx for idx, token in enumerate(vocab)}

# 변경된 인덱스를 토큰으로 복구하는 dict
id_to_token = {idx: token for idx, token in enumerate(vocab)}

# 단어 사전 내 단어 10개 출력 :
# ['<unk>', '.', '이', '영화', '의', '..', '가', '에', '...', '을']
print(vocab[:10])
# 단어사전 개수 출력 :
# 5001 (n_vocab + special_token 개수)
print(len(vocab))


# tokens 를 skip-gram 모델 입력용 데이터로 가공
# pair 리스트가 반환 되며, 첫번째는 중심 단어(입력값), 두번째는 생성 되어야 하는 주변 단어(정답값)가 됩니다.
# window_size 는 중심 단어에 대해 출력 해야 할 단어를 주변 몇번째 까지로 할지에 대한 값 입니다.
def get_word_pairs(tokens, window_size):
    pairs = []
    for sentence in tokens:
        sentence_length = len(sentence)
        for idx, center_word in enumerate(sentence):
            window_start = max(0, idx - window_size)
            window_end = min(sentence_length, idx + window_size + 1)
            center_word = sentence[idx]
            context_words = sentence[window_start:idx] + sentence[idx + 1:window_end]
            for context_word in context_words:
                pairs.append([center_word, context_word])
    return pairs


word_pairs = get_word_pairs(tokens, window_size=2)
# 출력 :
# [
#   ['굳', 'ㅋ'],
#   ['ㅋ', '굳'],
#   ['뭐', '야'],
#   ['뭐', '이'],
#   ['야', '뭐']
# ]
print(word_pairs[:5])


# skip-gram 모델 입력용 데이터를 인덱스로 변경
def get_index_pairs(word_pairs, token_to_id):
    pairs = []
    unk_index = token_to_id["<unk>"]
    for word_pair in word_pairs:
        center_word, context_word = word_pair
        center_index = token_to_id.get(center_word, unk_index)
        context_index = token_to_id.get(context_word, unk_index)
        pairs.append([center_index, context_index])
    return pairs


index_pairs = get_index_pairs(word_pairs, token_to_id)
# 인덱스 화 된 데이터 5 개 출력 :
# [
#   [595, 100],
#   [100, 595],
#   [77, 176],
#   [77, 2],
#   [176, 77]
# ]
print(index_pairs[:5])
# 5001
print(len(vocab))

# 데이터셋 정리
# (N, 2) 형태의 입력, 정답 pair 리스트를 텐서에 입력
index_pairs = torch.tensor(index_pairs)

# 중심 단어 리스트 분리
center_indexs = index_pairs[:, 0]
# tensor([595, 100,  77,  77, 176])
print(center_indexs[:5])

# 주변 단어 리스트 분리
contenxt_indexs = index_pairs[:, 1]
# tensor([100, 595, 176,   2,  77])
print(contenxt_indexs[:5])

# 중심 단어와 주변 단어 리스트로 데이터셋 준비 후 데이터 로더로 래핑
dataset = TensorDataset(center_indexs, contenxt_indexs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 사용 디바이스 설정 가져오기
device = tu.get_gpu_support_device(gpu_support=True)

# 기본 Word2Vec SkipGram 모델 생성
word2vec = VanillaSkipgram(vocab_size=len(token_to_id), embedding_dim=128).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(word2vec.parameters(), lr=0.1)

io_print_flag = True
for epoch in range(10):
    cost = 0.0
    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = word2vec(input_ids)
        loss = criterion(logits, target_ids)

        if io_print_flag:
            io_print_flag = False
            print("input")
            # tensor([   0,  209,    0, 1289,  262])
            print(input_ids[:5])

            print("output")
            # tensor([[ 0.2011,  1.0723, -0.3175,  ...,  0.6984,  0.1900,  0.4533],
            #         [ 0.3880, -0.2664,  0.2827,  ..., -0.5723, -0.3123,  0.4300],
            #         [ 0.2011,  1.0723, -0.3175,  ...,  0.6984,  0.1900,  0.4533],
            #         [ 0.1834,  0.6913,  0.0013,  ..., -0.1637, -0.6249,  0.8163],
            #         [-0.0508, -0.5118, -0.0853,  ..., -0.6565, -0.1343, -0.0602]],
            #        grad_fn=<SliceBackward0>)
            print(logits[:5])

            print("label")
            # tensor([105,  65,  19,  45, 966])
            print(target_ids[:5])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(dataloader)
    print(f"Epoch : {epoch + 1:4d}, Cost : {cost:.3f}")

# 학습된 모델의 임베딩 계층에서 단어에 대한 임베딩 리스트의 리스트(매트릭스)를 가져와서 넘파이로 대입
embedding_matrix = word2vec.embedding.weight.detach().cpu().numpy()

# 단어별 임베딩 벡터를 token_to_embedding 에 dict 타입(단어, 임베딩 리스트)으로 저장
token_to_embedding = dict()
for word, embedding in zip(vocab, embedding_matrix):
    token_to_embedding[word] = embedding

# 단어사전 인덱스 30 번 단어의 임베딩 리스트 가져와 출력
token = vocab[30]
token_embedding = token_to_embedding[token]
# ex : 연기
print(token)
# ex : [-0.8257925 1.3803437 ... 0.19158626 -0.74603665]
print(token_embedding)


# 임베딩 리스트는 단어의 의미를 지닌 벡터입니다.
# 고로 서로 비슷한 단어는 서로 비슷한 벡터를 지닐 것입니다.
# 이를 위하여 코사인 유사도를 측정하여 단어 임베딩 학습 결과를 확인할 것입니다.
def cosine_similarity(a, b):
    cosine = np.dot(b, a) / (norm(b, axis=1) * norm(a))
    return cosine


# 코사인 유사도에서 n 개를 추려오기
def top_n_index(cosine_matrix, n):
    closest_indexes = cosine_matrix.argsort()[::-1]
    top_n = closest_indexes[1: n + 1]
    return top_n


cosine_matrix = cosine_similarity(token_embedding, embedding_matrix)
top_n = top_n_index(cosine_matrix, n=5)

print(f"{token}와 가장 유사한 5 개 단어")
for index in top_n:
    print(f"{id_to_token[index]} - 유사도 : {cosine_matrix[index]:.4f}")
