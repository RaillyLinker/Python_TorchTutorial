from Korpora import Korpora
from gensim.models import FastText

# fastText 란, 2015 년 메타의 FAIR 연구소에서 개발한 오픈소스 임베딩 모델입니다.
# N-Gram 을 사용 하여 하위 단어를 고려 하며, Word2Vec 보다 더 높은 정확도와 성능을 제공한다고 합니다.
# 또한, Word2Vec 과는 달리 OOV 단어를 대상으로도 의미 있는 임베딩을 추출할 수 있다고 합니다.

# 한국어 자연어 이해 데이터셋(Korean Natural Language Inference) 말뭉치 로딩
# kornli 는, 자연어 추론을 위한 데이터 셋으로, 자연어 추론이란, 두개 이상의 문장이 주어졌을 때, 두 문장 간의 관계를 분류하는 작업을 의미합니다.
corpus = Korpora.load("kornli")
corpus_texts = corpus.get_all_texts() + corpus.get_all_pairs()

# 텍스트 데이터를 공백으로 나누기
# fastText 는, 입력 단어의 구조적 특징을 학습할 수 있으므로, 형태소 분리를 할 필요가 없다고 합니다.
tokens = [sentence.split() for sentence in corpus_texts]

# 텍스트 데이터 문장 3개 출력
# [
#   ['개념적으로', '크림', '스키밍은', '제품과', '지리라는', '두', '가지', '기본', '차원을', '가지고', '있다.'],
#   ['시즌', '중에', '알고', '있는', '거', '알아?', '네', '레벨에서', '다음', '레벨로', '잃어버리는', '거야', '브레이브스가',
#       '모팀을', '떠올리기로', '결정하면', '브레이브스가', '트리플', 'A에서', '한', '남자를', '떠올리기로', '결정하면', '더블', 'A가',
#       '그를', '대신하러', '올라가고', 'A', '한', '명이', '그를', '대신하러', '올라간다.'
#   ],
#   ['우리', '번호', '중', '하나가', '당신의', '지시를', '세밀하게', '수행할', '것이다.']
# ]
print(tokens[:3])

# FastText 모델 생성
fastText = FastText(
    sentences=tokens,
    # 임베딩 벡터 차원 수
    vector_size=128,
    # 학습 데이터 생성 윈도우 크기
    window=5,
    # 학습에 사용될 단어의 최소 빈도로, 이 값의 미만으로 나오는 단어는 학습에 사용하지 않습니다.
    min_count=5,
    # Skip-gram 모델 사용 여부 (0 : CBoW 사용, 1 : Skip-Gram 사용)
    sg=1,
    # 학습 에폭 수
    epochs=3,
    # N-Gram 최소값
    min_n=2,
    # N-Gram 최대값
    max_n=6
)

# 학습된 모델을 저장
fastText.save("../by_product_files/fastText.model")

# 모델 파일에서 불러오기
fastText = FastText.load("../by_product_files/fastText.model")

# OOV 테스트
oov_token = "사랑해요"
oov_vector = fastText.wv[oov_token]
# 아래 출력에서 보다시피, 단어 "사랑해요" 는 단어 사전에 없습니다.
print(oov_token in fastText.wv.index_to_key)
# 하지만 유의미한 벡터를 얻을 수 있고, 그에 가장 비슷한 단어들(사랑, 사랑에, 사랑의, 등...)을 가져올 수 있습니다.
print(fastText.wv.most_similar(oov_vector, topn=5))
