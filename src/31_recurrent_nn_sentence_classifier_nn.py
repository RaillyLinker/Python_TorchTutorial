import utils.torch_util as tu
import model_layers.recurrent_nn_sentence_classifier.main_model as recurrent_nn_sentence_classifier
import os
import torch
from torch import nn
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from collections import Counter
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec
from torch import optim

"""
[순환 신경망 모델 문장 분류기]
RNN, LSTM 을 이용한 문장 분류기를 만들 것입니다.
문장을 입력 하면, 해당 문장에 포함된 감정을 분석 하여 분류 하는 모델입니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    corpus = Korpora.load("nsmc")
    corpus_df = pd.DataFrame(corpus.test)

    train = corpus_df.sample(frac=0.9, random_state=42)
    test = corpus_df.drop(train.index)

    print(train.head(5).to_markdown())
    print("Training Data Size :", len(train))
    print("Testing Data Size :", len(test))

    def build_vocab(corpus, n_vocab, special_tokens):
        counter = Counter()
        for tokens in corpus:
            counter.update(tokens)
        vocab = special_tokens
        for token, count in counter.most_common(n_vocab):
            vocab.append(token)
        return vocab

    tokenizer = Okt()
    train_tokens = [tokenizer.morphs(review) for review in train.text]
    test_tokens = [tokenizer.morphs(review) for review in test.text]

    vocab = build_vocab(corpus=train_tokens, n_vocab=5000, special_tokens=["<pad>", "<unk>"])
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

    max_length = 32
    pad_id = token_to_id["<pad>"]
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

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 모델 생성
    n_vocab = len(token_to_id)
    hidden_dim = 64
    embedding_dim = 128
    n_layers = 2

    word2vec = Word2Vec.load("../_by_product_files/gensim_word_2_vec/word2vec.model")
    init_embeddings = np.zeros((n_vocab, embedding_dim))

    for index, token in id_to_token.items():
        if token not in ["<pad>", "<unk>"]:
            init_embeddings[index] = word2vec.wv[token]

    model = recurrent_nn_sentence_classifier.MainModel(
        n_vocab=n_vocab, hidden_dim=hidden_dim, embedding_dim=embedding_dim,
        n_layers=n_layers, pretrained_embedding=init_embeddings
    )

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optim.RMSprop(model.parameters(), lr=0.001),
        train_dataloader=train_dataloader,
        num_epochs=5,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/recurrent_nn_sentence_classifier",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/recurrent_nn_sentence_classifier"
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
