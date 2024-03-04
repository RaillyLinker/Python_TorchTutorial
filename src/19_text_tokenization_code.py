from jamo import h2j, j2hcj

# konlpy 는 JDK 를 설치하고 환경변수를 등록해야 합니다.
from konlpy.tag import Okt
from konlpy.tag import Kkma

# [텍스트 토큰화]
# 텍스트를 토큰화 한다는 것은, 텍스트를 분리하여 고유성을 부과하는 것을 의미합니다.
# 자연어 분석시 소설 책의 내용을 분석한다고 해봅시다.
# 이때 분석 모델에 소설책의 모든 글자를 한번에 전부 입력해야할까요?
# 그렇게 된다면 입력 데이터가 비정형이 되며, 입력될 데이터 크기의 가능성은 무한해질 것입니다.
# 사람의 경우를 생각해봅시다.
# 사람이 소설을 보고 그 내용을 분석한다고 할 때,
# 책 전체 내용을 한번에 뇌에 넣어버리지는 않습니다.
# 한 페이지도 아니고, 심지어 한 줄을 그대로 넣기도 힘듭니다.
# 결국 언어라는 것을 구성하는 최소 단위를 순서대로 입력하고 또 입력해야 합니다.
# 이렇게 자연어를 특정 기준에 따라 쪼개는 것을 토큰화라고 부릅니다.
# 아래 코드는 텍스트 데이터를 토큰화 하는 기본적인 방식을 몇가지 구현한 코드입니다.

review_text = " 현실과 구분 불가능한 cg. 시각적 즐거음은 최고! 더불어 ost는 더더욱 최고!!"

# (1. 공백으로 나누기)
# 공백으로 나누기는, 가장 단순한 자연어 쪼개기 방법입니다.
# 단점으로는 오탈자, 문장부호, 띄어쓰기 오류 등에 취약합니다.
tokenized = review_text.split()
print("공백으로 나누기")
print(tokenized)
print("")

# (글자 나누기)
# 글자를 하나 하나 나눕니다.
# 공백으로 나누는 것은 글자의 조합에 따라 무한히 많은 형태가 될 수 있지만, 이 방식은 고유성을 보장 합니다.
# 언어의 의미가 아닌 글자 부호의 고유성에 대한 정보 만을 지니는 것을 주의
# 아래 방식은 가장 단순한 방식으로, 한글의 경우는 자모가 결합된 상태로 반환이 됩니다.
tokenized = list(review_text)
print("글자 나누기")
print(tokenized)
print("")

# (한글 글자 자모 나누기)
# 글자 나누기와 동일한 방식인데, 한글에서 자모까지 분리합니다.
decomposed = j2hcj(h2j(review_text))
tokenized = list(decomposed)
print("한글 글자 자모 나누기")
print(tokenized)
print("")

# (형태소로 나누기)
# 의미를 지닌 가장 작은 단위로 나누는 방식입니다.
# 알고리즘으로 문장에서 형태소를 분리해야하는데, 라이브러리가 여럿 존재합니다.

# (형태소로 나누기 - OKT)
# SNS 텍스트 데이터 기반으로 개발된 알고리즘 입니다.
okt = Okt()

nouns = okt.nouns(review_text)
phrases = okt.phrases(review_text)
morphs = okt.morphs(review_text)
pos = okt.pos(review_text)
print("형태소로 나누기 - OKT")
print("명사 추출 :", nouns)
print("구 추출 :", phrases)
print("형태소 추출 :", morphs)
print("품사 태깅 :", pos)
print("")

# (형태소로 나누기 - KKma)
# 국립국어원에서 배포한 세종 말뭉치 데이터 기반으로 개발된 알고리즘 입니다.
kkma = Kkma()

nouns = kkma.nouns(review_text)
sentences = kkma.sentences(review_text)
morphs = kkma.morphs(review_text)
pos = kkma.pos(review_text)
print("형태소로 나누기 - KKma")
print("명사 추출 :", nouns)
print("문장 추출 :", sentences)
print("형태소 추출 :", morphs)
print("품사 태깅 :", pos)
print("")

# (형태소 분리의 단점)
# 형태소 분리의 취약점은, 형태소 분리의 주체가 형태소 분리기라는 것이고,
# 즉 기존에 준비된 형태소 이외에 신조어, 오탈자, 축약어 등에 취약할 수 있다는 것입니다.
# 이에 대한 해결 방법중 하나는, 하위 단어 토큰화(Subword Tokenization) 이 있습니다.
# 예를들어 Reinforcement 와 같은 단어가 있다하면 이를 Rein, Force, ment 와 같이 하위단어로 나누어서 처리하는 것입니다.
# 하위 단어 토큰화 방식에는 바이트 페어 인코딩(빈도 기반), 워드피스(확률 기반), 유니그램 모델 등이 있습니다.
