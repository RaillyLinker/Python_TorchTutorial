import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

"""
[기본 텍스트 임베딩 정리]
텍스트 토크나이징은 텍스트를 작은 단위로 분리하고 고유성을 부여하는 작업입니다.
텍스트 임베딩은 분리된 텍스트에 의미를 부여하여 자연어 처리에 이용하기 위해 전처리 하는 작업이라 생각하세요.
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

print(unigram)
print(bigram)
print(trigram)

# NLTK 에서 제공하는 N-gram
unigram = nltk.ngrams(sentence.split(), 1)
bigram = nltk.ngrams(sentence.split(), 2)
trigram = nltk.ngrams(sentence.split(), 3)

print(list(unigram))
print(list(bigram))
print(list(trigram))

# [TF-IDF]
# TF(Term Frequency)
# : 단어 빈도를 의미합니다.
# 문서 내에서 특정 단어의 빈도수를 나타내는 값으로,
# 예를들어 문서 내에 'movie' 라는 단어가 4 번 등장하면, 해당 단어의 TF 는 4 입니다.

# DF(Document Frequency)
# : 문서 빈도를 의미합니다.
# 총 데이터 내 모든 문서를 범위로 하였을 때,
# 특정 단어의 출현이 몇개의 문서에서 등장하는지에 대한 횟수를 의미합니다.
# 예를들어 'movie' 라는 단어가 특정 문서에서 4 번 등장했고, 이 단어가 등장하는 문서의 개수가 5개 라고 하면,
# DF 는 5 라고 할 수 있습니다.

# IDF(Inverse Document Frequency)
# : 역 문서 빈도를 의미합니다.
# 전체 문서 수를 문서 빈도(DF)로 나누어서 로그를 취한 값입니다.
# 수식, "전체 문서 수 / 1 + 문서 빈도" 의 의미는, 해당 단어가 문서 빈도가 작을수록 큰 의미를 지닌다는 뜻으로,
# 예를들어 따옴표나 느낌표와 같이 일상적으로 등장하는 단어는 문장의 주제에 영향을 끼치기보다는
# 문장의 형태를 구성하는 역할만을 하기에 그 중요도를 낮게 해야하기 때문입니다.
# 문서 빈도에 1을 더한 것은, DF 가 0 일때 에러가 나지 않게 하기 위한 것이고,
# 로그를 취했다는 것은  전체 문서 수가 커짐에 따라 너무 큰 값이 반환되는 것을 막기 위한 것입니다.

# TF-IDF(Term Frequency - Inverse Document Frequency)
# 앞서 TF 에 IDF 를 곱한 것입니다.
# IDF 가 높다는 것은 해당 단어가 의미 없이 일반적으로 등장하는 단어가 아니라는 것이고,
# TF 가 높다는 것은 해당 단어가 해당 문서 내에서 큰 역할을 한다는 것을 의미합니다.
# 즉, 일반적이지 않고, 특정 문서에 큰 영향을 끼치는 단어라는 것을 의미하는 수치입니다.

# TF-IDF 벡터화
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

print(tfidf_matrix.toarray())
print(tfidf_vectorizer.vocabulary_)

# 위 결과에서 사전은,
# {'actor': 0, 'don': 1, 'famous': 2, 'is': 3, 'like': 4, 'movie': 5, 'that': 6}
# 이러한데, "That movie is famous movie" 이것을 벡터화하면,
# 위 사전과 동일한 사이즈의 벡터를 [0, 0, 0, 0, 0, 0, 0] 이렇게 준비해두고, 사전 내 각 단어들에 대한 TF-IDF 를 계산하면 됩니다.
# actor 는 없으니 0, don 도 없으니 0, famous, is 를 계산하고, like 도 없으니 0, movie 와 that 역시 계산하면,
# "That movie is famous movie" 에 대한 TF-IDF 의 계산값은
# [0. 0. 0.39687454 0.39687454 0. 0.79374908 0.2344005]
# 이렇게 됩니다.
# 위 벡터는 문장에 등장하는 모든 단어들에 대한 빈도의 통계값으로 문서의 의미를 나타낸 것으로 해석할 수 있습니다.

# 이 방식의 임베딩 결과는 문장의 순서나 문맥을 고려하지 않으며,
# 벡터가 각 단어별 문장 내에서의 중요도를 의미하기만 할 뿐 단어의 의미를 가지고 있지는 않다는 것을 주의하세요.
