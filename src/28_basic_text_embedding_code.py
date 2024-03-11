import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import pandas as pd  # 데이터프레임 사용을 위해
from math import log  # IDF 계산을 위해

"""
[기본 텍스트 임베딩 정리]
- 텍스트 토크나이징은 텍스트를 작은 단위로 분리하고 고유성을 부여하는 작업입니다.
    텍스트 임베딩은 분리된 텍스트에 의미를 부여하여 자연어 처리에 이용하기 위해 전처리 하는 작업이라 생각하세요.
- 예를 들어보겠습니다.
    단어 '사과'를 수치화하면 어떻게 해야할까요?
    사과에 관련된 모든 기준을 만들어 점수를 매기면 됩니다.
    예를들어 빨간 정도, 맛있는 정도, 평균 가격, 유동기한 과 같은 데이터를 준비했다고 합시다.
    이를 벡터로 나타내면, [1, 0.4, 0.2, 0.6] 이렇게 점수를 매길 수 있다고 합시다.
    동일한 기준으로 '바나나' 라는 단어를 수치화 한다고 해봅시다. 그외에도 키위, 포도와 같은 데이터를 모두 나타냈을 때,
    이때 사과라는 것을 나타낼 수 있는 고유한 벡터를 가진다고 한다면 이는 사과를 훌륭하게 수치화 했다고 할 수 있습니다.
    반대로, 위와 같은 기준으로 점수를 매겼을 때, 사과에서 나온 데이터의 패턴과 유사하게 나타나는 단어는 사과 혹은 그에 유사한 단어라고
    판단을 내릴 수도 있을 것입니다.
    위의 예시는 매우 빈약한데, 만약 이러한 정보가 모두 모이면 '사과' 라는 하나의 단어를 표현하는 정보의 무더기라 할 수 있겠네요.
    머신러닝에서 텍스트 임베딩이 해야하는 것도 이러한 것과 같습니다.
    단순한 텍스트 단어를 수치화 하여 벡터로 나타내는 것이죠.
"""


# [N-gram]
# 텍스트 토큰을 n개로 묶어서 반환하는 방법입니다.

# 함수 직접 구현
def ngrams(sentence, n):
    words = sentence.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return list(ngrams)


sentence = "안녕하세요 만나서 진심으로 반가워요"

unigram = ngrams(sentence, 1)
bigram = ngrams(sentence, 2)
trigram = ngrams(sentence, 3)

# 출력 : [('안녕하세요',), ('만나서',), ('진심으로',), ('반가워요',)]
print(unigram)
# 출력 : [('안녕하세요', '만나서'), ('만나서', '진심으로'), ('진심으로', '반가워요')]
print(bigram)
# 출력 : [('안녕하세요', '만나서', '진심으로'), ('만나서', '진심으로', '반가워요')]
print(trigram)

# NLTK 에서 제공하는 N-gram
unigram = nltk.ngrams(sentence.split(), 1)
bigram = nltk.ngrams(sentence.split(), 2)
trigram = nltk.ngrams(sentence.split(), 3)

# 출력 : [('안녕하세요',), ('만나서',), ('진심으로',), ('반가워요',)]
print(list(unigram))
# 출력 : [('안녕하세요', '만나서'), ('만나서', '진심으로'), ('진심으로', '반가워요')]
print(list(bigram))
# 출력 : [('안녕하세요', '만나서', '진심으로'), ('만나서', '진심으로', '반가워요')]
print(list(trigram))

"""
[Bag of Words]
- Bag of Words란 단어들의 순서는 전혀 고려하지 않고, 
    단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법입니다. 
    Bag of Words를 직역하면 단어들의 가방이라는 의미입니다. 
    단어들이 들어있는 가방을 상상해봅시다. 
    갖고있는 어떤 텍스트 문서에 있는 단어들을 가방에다가 전부 넣습니다. 
    그 후에는 이 가방을 흔들어 단어들을 섞습니다. 
    만약, 해당 문서 내에서 특정 단어가 N번 등장했다면, 이 가방에는 그 특정 단어가 N개 있게됩니다. 
    또한 가방을 흔들어서 단어를 섞었기 때문에 더 이상 단어의 순서는 중요하지 않습니다.
    
- BoW를 만드는 과정
    1. 각 단어에 고유한 정수 인덱스를 부여합니다. (단어 집합 생성)
    2. 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터를 만듭니다.  
"""


# BoW 구현 함수
def build_bag_of_words(document):
    okt = Okt()
    # 온점 제거 및 형태소 분석
    document = document.replace('.', '')
    tokenized_document = okt.morphs(document)

    word_to_index = {}
    bow = []

    for word in tokenized_document:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)
            # BoW에 전부 기본값 1을 넣는다.
            bow.insert(len(word_to_index) - 1, 1)
        else:
            # 재등장하는 단어의 인덱스
            index = word_to_index.get(word)
            # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
            bow[index] = bow[index] + 1

    return word_to_index, bow


doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
vocab, bow = build_bag_of_words(doc1)
# 출력 : vocabulary : {'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9}
print('vocabulary :', vocab)
# 출력 : bag of words vector : [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
print('bag of words vector :', bow)

"""
- 위에서 보는 것처럼, BoW 는 문장 내에서 vocabulary 의 단어가 몇번 등장했는지를 표현하는 벡터를 반환합니다.

- BoW는 각 단어가 등장한 횟수를 수치화하는 텍스트 표현 방법이므로 
    주로 어떤 단어가 얼마나 등장했는지를 기준으로 문서가 어떤 성격의 문서인지를 판단하는 작업에 쓰입니다. 
    즉, 분류 문제나 여러 문서 간의 유사도를 구하는 문제에 주로 쓰입니다. 
    가령, '달리기', '체력', '근력'과 같은 단어가 자주 등장하면 해당 문서를 체육 관련 문서로 분류할 수 있을 것이며, 
    '미분', '방정식', '부등식'과 같은 단어가 자주 등장한다면 수학 관련 문서로 분류할 수 있습니다.
"""

"""
[문서의 벡터화: 문서 단어 행렬(Document-Term Matrix, DTM)]
- 서로 다른 문서들의 BoW들을 결합한 표현 방법인 문서 단어 행렬(Document-Term Matrix, DTM) 표현 방법을 배워보겠습니다. 
    행과 열을 반대로 선택하면 TDM이라고 부르기도 합니다. 
    이렇게 하면 서로 다른 문서들을 비교할 수 있게 됩니다.

- 문서 단어 행렬(Document-Term Matrix, DTM)의 표기법
    문서 단어 행렬(Document-Term Matrix, DTM)이란 다수의 문서에서 등장하는 각 단어들의 빈도를 행렬로 표현한 것을 말합니다. 
    쉽게 생각하면 각 문서에 대한 BoW를 하나의 행렬로 만든 것으로 생각할 수 있으며, 
    BoW와 다른 표현 방법이 아니라 BoW 표현을 다수의 문서에 대해서 행렬로 표현하고 부르는 용어입니다. 
    예를 들어서 이렇게 4개의 문서가 있다고 합시다.

    문서1 : 먹고 싶은 사과
    문서2 : 먹고 싶은 바나나
    문서3 : 길고 노란 바나나 바나나
    문서4 : 저는 과일이 좋아요
    
    띄어쓰기 단위 토큰화를 수행한다고 가정하고, 문서 단어 행렬로 표현하면 다음과 같습니다.
    
        과일이 길고  노란  먹고  바나나 사과  싶은  저는  좋아요
    문서1 0    0    0    1    0    1    1    0    0
    문서2 0    0    0    1    1    0    1    0    0
    문서3 0    1    1    0    2    0    0    0    0
    문서4 1    0    0    0    0    0    0    1    1
    
    보다시피, 먼저 모든 문서를 이용하여 vocabulary 를 만들고, 
    그것을 통해 각 문서별 cbow 를 구해서 행렬 형식으로 저장하면 됩니다.
    
- 문서 단어 행렬(Document-Term Matrix)의 한계
    DTM은 매우 간단하고 구현하기도 쉽지만, 본질적으로 가지는 몇 가지 한계들이 있습니다.
    
    1) 희소 표현(Sparse representation)
        원-핫 벡터는 단어 집합의 크기가 벡터의 차원이 되고 대부분의 값이 0이 되는 벡터입니다. 
        원-핫 벡터는 공간적 낭비와 계산 리소스를 증가시킬 수 있다는 점에서 단점을 가집니다. 
        DTM도 마찬가지입니다. DTM에서의 각 행을 문서 벡터라고 해봅시다. 
        각 문서 벡터의 차원은 원-핫 벡터와 마찬가지로 전체 단어 집합의 크기를 가집니다. 
        만약 가지고 있는 전체 코퍼스가 방대한 데이터라면 문서 벡터의 차원은 수만 이상의 차원을 가질 수도 있습니다. 
        또한 많은 문서 벡터가 대부분의 값이 0을 가질 수도 있습니다. 
        당장 위에서 예로 들었던 문서 단어 행렬의 모든 행이 0이 아닌 값보다 0의 값이 더 많은 것을 볼 수 있습니다.
        원-핫 벡터나 DTM과 같은 대부분의 값이 0인 표현을 희소 벡터(sparse vector) 또는 희소 행렬(sparse matrix)라고 부르는데, 
        희소 벡터는 많은 양의 저장 공간과 높은 계산 복잡도를 요구합니다. 
        이러한 이유로 전처리를 통해 단어 집합의 크기를 줄이는 일은 BoW 표현을 사용하는 모델에서 중요할 수 있습니다. 
        텍스트 전처리 방법을 사용하여 구두점, 빈도수가 낮은 단어, 불용어를 제거하고, 
        어간이나 표제어 추출을 통해 단어를 정규화하여 단어 집합의 크기를 줄일 수 있습니다.
    
    2) 단순 빈도 수 기반 접근
        여러 문서에 등장하는 모든 단어에 대해서 빈도 표기를 하는 이런 방법은 때로는 한계를 가지기도 합니다. 
        예를 들어 영어에 대해서 DTM을 만들었을 때, 불용어인 the는 어떤 문서이든 자주 등장할 수 밖에 없습니다. 
        그런데 유사한 문서인지 비교하고 싶은 문서1, 문서2, 문서3에서 
        동일하게 the가 빈도수가 높다고 해서 이 문서들이 유사한 문서라고 판단해서는 안 됩니다.
        각 문서에는 중요한 단어와 불필요한 단어들이 혼재되어 있습니다. 
        앞서 불용어(stopwords)와 같은 단어들은 빈도수가 높더라도 자연어 처리에 있어 의미를 갖지 못하는 단어라고 언급한 바 있습니다. 
        그렇다면 DTM에 불용어와 중요한 단어에 대해서 가중치를 줄 수 있는 방법은 없을까요? 
        이러한 아이디어를 적용한 TF-IDF를 이어서 정리합니다.
"""

"""
[TF-IDF(Term Frequency-Inverse Document Frequency)]
- TF(Term Frequency)
    단어 빈도를 의미합니다.
    문서 내에서 특정 단어의 빈도수를 나타내는 값으로,
    예를들어 문서 내에 'movie' 라는 단어가 4 번 등장하면, 해당 단어의 TF 는 4 입니다.

- DF(Document Frequency)
    문서 빈도를 의미합니다.
    총 데이터 내 모든 문서를 범위로 하였을 때,
    특정 단어의 출현이 몇개의 문서에서 등장하는지에 대한 횟수를 의미합니다.
    예를들어 'movie' 라는 단어가 특정 문서에서 4 번 등장했고, 이 단어가 등장하는 문서의 개수가 5개 라고 하면,
    DF 는 5 라고 할 수 있습니다.

- IDF(Inverse Document Frequency)
    역 문서 빈도를 의미합니다.
    전체 문서 수를 문서 빈도(DF)로 나누어서 로그를 취한 값입니다.
    수식, "전체 문서 수 / 1 + 문서 빈도" 의 의미는, 해당 단어가 문서 빈도가 작을수록 큰 의미를 지닌다는 뜻으로,
    예를들어 따옴표나 느낌표와 같이 일상적으로 등장하는 단어는 문장의 주제에 영향을 끼치기보다는
    문장의 형태를 구성하는 역할만을 하기에 그 중요도를 낮게 해야하기 때문입니다.
    문서 빈도에 1을 더한 것은, DF 가 0 일때 에러가 나지 않게 하기 위한 것이고,
    로그를 취했다는 것은  전체 문서 수가 커짐에 따라 너무 큰 값이 반환되는 것을 막기 위한 것입니다.

- TF-IDF(Term Frequency - Inverse Document Frequency)
    앞서 TF 에 IDF 를 곱한 것입니다.
    IDF 가 높다는 것은 해당 단어가 의미 없이 일반적으로 등장하는 단어가 아니라는 것이고,
    TF 가 높다는 것은 해당 단어가 해당 문서 내에서 큰 역할을 한다는 것을 의미합니다.
    즉, 일반적이지 않고, 특정 문서에 큰 영향을 끼치는 단어라는 것을 의미하는 수치입니다.
"""
# (TF-IDF 직접 구현)
docs = [
    '먹고 싶은 사과',
    '먹고 싶은 바나나',
    '길고 노란 바나나 바나나',
    '저는 과일이 좋아요'
]

# 1. 단어 사전 생성
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

N = len(docs)  # 총 문서의 수


def tf(t, d):
    return d.count(t)


def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N / (df + 1))


def tfidf(t, d):
    return tf(t, d) * idf(t)


result = []

# 각 문서에 대해서 BoW 와 동일하게 반복
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns=vocab)
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
# 출력 :
#           IDF
# 과일이  0.693147
# 길고   0.693147
# 노란   0.693147
# 먹고   0.287682
# 바나나  0.287682
# 사과   0.693147
# 싶은   0.287682
# 저는   0.693147
# 좋아요  0.693147
print(idf_)
result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d))

tfidf_ = pd.DataFrame(result, columns=vocab)
# 출력 :
#         과일이        길고        노란  ...        싶은        저는       좋아요
# 0  0.000000  0.000000  0.000000  ...  0.287682  0.000000  0.000000
# 1  0.000000  0.000000  0.000000  ...  0.287682  0.000000  0.000000
# 2  0.000000  0.693147  0.693147  ...  0.000000  0.000000  0.000000
# 3  0.693147  0.000000  0.000000  ...  0.000000  0.693147  0.693147
print(tfidf_)

# (TfidfVectorizer 를 사용한 TF-IDF 벡터화)
# TF-IDF 를 사용하여 문장을 벡터화합니다.
corpus = [
    "That movie is famous movie",
    "I like that actor",
    "I don’t like that actor"
]
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(corpus)
tfidf_matrix = tfidf_vectorizer.transform(corpus)
# tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# 출력 :
# [
#   [0. 0. 0.39687454 0.39687454 0. 0.79374908 0.2344005 ]
#   [0.61980538 0. 0. 0. 0.61980538 0. 0.48133417]
#   [0.4804584  0.63174505 0. 0. 0.4804584  0. 0.37311881]
# ]
print(tfidf_matrix.toarray())
# 출력 : {'that': 6, 'movie': 5, 'is': 3, 'famous': 2, 'like': 4, 'actor': 0, 'don': 1}
print(tfidf_vectorizer.vocabulary_)

"""
- 위 결과에서 사전은,
    {'actor': 0, 'don': 1, 'famous': 2, 'is': 3, 'like': 4, 'movie': 5, 'that': 6}
    이러한데, "That movie is famous movie" 이것을 벡터화하면,
    위 사전과 동일한 사이즈의 벡터를 [0, 0, 0, 0, 0, 0, 0] 이렇게 준비해두고, 사전 내 각 단어들에 대한 TF-IDF 를 계산하면 됩니다.
    actor 는 없으니 0, don 도 없으니 0, famous, is 를 계산하고, like 도 없으니 0, movie 와 that 역시 계산하면,
    "That movie is famous movie" 에 대한 TF-IDF 의 계산값은
    [0. 0. 0.39687454 0.39687454 0. 0.79374908 0.2344005]
    이렇게 됩니다.
    위 벡터는 문장에 등장하는 모든 단어들에 대한 빈도의 통계값으로 문서의 의미를 나타낸 것으로 해석할 수 있습니다.

- 이 방식의 임베딩 결과는 문장의 순서나 문맥을 고려하지 않으며,
    벡터가 각 단어별 문장 내에서의 중요도를 의미하기만 할 뿐 단어의 의미를 가지고 있지는 않다는 것을 주의하세요.
    
- 결과적으로, DTM 에서 단순 빈도 표현에서 나타나는 단어 중요도 미포함 문제는 해결되었는데,
    여전히 희소 벡터 문제는 존재합니다.
"""
