from jamo import h2j, j2hcj

# konlpy 는 JDK 를 설치하고 환경변수를 등록해야 합니다.
from konlpy.tag import Okt
from konlpy.tag import Kkma

review_text = " 현실과 구분 불가능한 cg. 시각적 즐거음은 최고! 더불어 ost는 더더욱 최고!!"

# (공백으로 나누기)
# 공백으로 나누기는, 가장 단순하며, 오탈자, 문장부호, 띄어쓰기 오류 등에 취약합니다.
tokenized = review_text.split()
print("공백으로 나누기")
print(tokenized)
print("")

# (글자 나누기)
# 글자 하나하나로 나눕니다.
# 공백으로 나누는 것은 글자의 조합에 따라 무한히 많은 형태가 될 수 있지만, 이 방식은 고유성을 보장합니다.
# 언어의 의미가 아닌 글자 부호의 고유성에 대한 정보만을 지니는 것을 주의
# 아래 방식은 가장 단순한 방식으로, 한글의 경우는 자모가 결합된 상태로 반환이 됩니다.
tokenized = list(review_text)
print("글자 나누기")
print(tokenized)
print("")

# (한글 글자 자모 나누기)
# 한글에서 자모까지 분리합니다.
decomposed = j2hcj(h2j(review_text))
tokenized = list(decomposed)
print("한글 글자 자모 나누기")
print(tokenized)
print("")

# (형태소로 나누기)
# 의미를 지닌 가장 작은 단위로 나누는 방식입니다.
# 알고리즘으로 문장에서 형태소를 분리해야하는데, 라이브러리가 여럿 존재합니다.
# todo 단어 의미를 latent vector 로 자동으로 학습하도록 신경망 설계를 해볼것

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
