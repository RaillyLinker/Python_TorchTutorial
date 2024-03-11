import utils.torch_util as tu
import pandas as pd
from Korpora import Korpora
import torch
from konlpy.tag import Okt
from collections import Counter
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import model_layers.cnn_sentence_classifier.main_model as cnn_sentence_classifier
from gensim.models import Word2Vec
from torch import nn
from torch import optim
import os

"""
[자연어 처리 합성곱 신경망 적용 샘플]
이 샘플은 일반적으로 컴퓨터 비전에 사용되는 합성곱 신경망을 자연어 처리에 이용 하는 예시를 보여줍니다.

원리를 설명합니다.
임베딩을 거쳐서 단어는 벡터가 됩니다.
단어 벡터가 모이면 문장 메트릭스가 됩니다.
CNN 을 이용하여 문장 메트릭스 내에서 단어 백터 N 개에 대하여 합성곱을 진행(1 차원 합성곱)하며 나아가면,
단어 N 개에 대한 의미가 추출되게 됩니다.
그렇게 단어 N 개에 대한 의미를 나타내는 투사 벡터가 여럿 모이면 해당 문장의 의미를 품은 것과 같게 되는 것입니다.
이는 단어 하나하나가 아니라 단어 N 개에 대해 서로간의 연결관계를 모델에 적용이 가능하다는 것입니다.

개인적으로는 문장 전체에서 단어 하나에 대한 어텐션을 구하는 방식보다는 효과적이지 못한 방식이라 생각이 됩니다.
그럼에도 이 방식이 의의가 있는 것이, 직렬적으로 한 단어씩 진행할 수 밖에 없는 선천적인 약점을 지닌 RNN 모델을 사용하지 않고
문장 내 단어의 배치라는 위치적인 정보를 사용하면서 보다 효율적으로 진행 할 수 있다는 것이 놀라운 모델입니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    corpus = Korpora.load("nsmc")
    train = pd.DataFrame(corpus.train)
    test = pd.DataFrame(corpus.test)
    print(train.head(5).to_markdown())
    print("Training Data Size :", len(train))
    print("Testing Data Size :", len(test))

    tokenizer = Okt()
    train_tokens = [tokenizer.morphs(review) for review in train.text]
    test_tokens = [tokenizer.morphs(review) for review in test.text]

    counter = Counter()
    for tokens in train_tokens:
        counter.update(tokens)
    vocab = ["<pad>", "<unk>"]
    for token, count in counter.most_common(5000):
        vocab.append(token)
    vocab = vocab
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for idx, token in enumerate(vocab)}

    print(vocab[:10])
    print(len(vocab))

    def pad_sequences(sequences, max_length, pad_value):
        result = list()
        for sequence in sequences:
            sequence = sequence[:max_length]
            pad_length = max_length - len(sequence)
            padded_sequence = sequence + [pad_value] * pad_length
            result.append(padded_sequence)
        return np.asarray(result)

    unk_id = token_to_id["<unk>"]
    train_ids = [
        [token_to_id.get(token, unk_id) for token in review] for review in train_tokens
    ]
    test_ids = [
        [token_to_id.get(token, unk_id) for token in review] for review in test_tokens
    ]

    pad_id = token_to_id["<pad>"]
    max_length = 32
    train_ids = pad_sequences(train_ids, max_length, pad_id)
    test_ids = pad_sequences(test_ids, max_length, pad_id)

    print(train_ids[0])
    print(test_ids[0])

    train_ids = torch.tensor(train_ids)
    test_ids = torch.tensor(test_ids)

    train_labels = torch.tensor(train.label.values, dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(test.label.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(train_ids, train_labels)
    test_dataset = TensorDataset(test_ids, test_labels)

    # 데이터 로더 래핑
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    n_vocab = len(token_to_id)
    embedding_dim = 128
    init_embeddings = np.zeros((n_vocab, embedding_dim))
    word2vec = Word2Vec.load("../_by_product_files/gensim_word_2_vec/word2vec.model")
    for index, token in id_to_token.items():
        if token not in ["<pad>", "<unk>"]:
            init_embeddings[index] = word2vec.wv[token]

    # 모델 생성
    model = cnn_sentence_classifier.MainModel(
        pretrained_embedding=init_embeddings,
        max_length=max_length
    )

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optim.RMSprop(model.parameters(), lr=0.0001),
        train_dataloader=train_dataloader,
        num_epochs=5,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/cnn_sentence_classifier",
        # check_point_load_file_full_path="../check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/cnn_sentence_classifier"
    if not os.path.exists(model_file_save_directory_path):
        os.makedirs(model_file_save_directory_path)
    save_file_full_path = tu.save_model_file(
        model=model,
        model_file_save_directory_path=model_file_save_directory_path
    )

    # # 저장된 모델 불러오기
    model = torch.load(save_file_full_path, map_location=device)
    print("Model Load Complete!")
    print(model)


if __name__ == '__main__':
    main()
