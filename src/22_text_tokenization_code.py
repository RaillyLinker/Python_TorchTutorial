from jamo import h2j, j2hcj

# konlpy 는 JDK 를 설치하고 환경변수를 등록해야 합니다.
from konlpy.tag import Okt
from konlpy.tag import Kkma

"""
[텍스트 토큰화]
- 텍스트를 토큰화 한다는 것은, 연속적 데이터인 텍스트를 일정한 기준에 따라 분리하여 고유성을 부과하는 것을 의미합니다. (자연어 분석의 최소 단위로 쪼개기)
    자연어 분석시 소설 책의 내용을 분석한다고 해봅시다.
    이때 분석 모델에 소설책의 모든 글자를 한번에 전부 입력 해야 할까요?
    그렇게 된다면 입력 데이터가 비정형이 되며, 입력될 데이터 크기는 무한해질 수 있습니다.
    그렇다면 글자를 쪼개서 하나하나 입력을 해야 할까요?
    아니라면 단어 사전에 있는 글자로 나누어 입력을 하는 방식이나, 더 단순히 띄어쓰기 단위로 나누어 입력하는 방식도 있을 수 있습니다.
    결국 언어라는 것을 구성하는 최소 단위를 순서대로 입력하고 또 입력해야 합니다.
    이렇게 자연어를 특정 기준에 따라 쪼개는 것을 토큰화라고 부릅니다.
    즉, 이렇게 쪼개어져서 입력되는 자연어의 단위를 토큰이라 부르는 것입니다.

- 토큰화 고려사항
    1. 구두점이나 특수 문자를 단순 제외해서는 안 됩니다.
    예를들어, m.p.h나 Ph.D나 AT&T, $45.55 와 같이 기호가 붙어야 의미를 가지는 단어가 존재합니다.
    
    2. 줄임말과 단어 내에 띄어쓰기가 있는 경우.
    예를 들어 what're는 what are의 줄임말이며, we're는 we are의 줄임말입니다. 
    또한 New York이라는 단어나 rock 'n' roll이라는 단어를 봅시다. 
    이 단어들은 하나의 단어이지만 중간에 띄어쓰기가 존재합니다. 
    토큰화 작업은 저러한 단어를 하나로 인식할 수 있는 능력도 가져야합니다.
    
    3. 영어에서 home-based 와 같이 하이픈으로 연결되어야 온전한 단어는 하나로 유지하고,
     doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 does 와 n`t 와 같이 분리해줍니다.
     이러한 언어적 특징을 고려해야합니다.
     
    4. 품사 태깅이 필요할 수 있습니다.
    예를들어 영어 단어 fly 는 동사로는 날다 라는 의미이지만, 명사로는 파리라는 의미가 있을 수 있습니다.
    이처럼 품사를 알아야 단어의 의미가 완성되는 경우가 있습니다.
 
- 한국어 토큰화의 어려움
    1. 교착어의 특성
    영어와는 달리 한국어에는 조사라는 것이 존재합니다.
    예를 들어 한국어에 그(he/him)라는 주어나 목적어가 들어간 문장이 있다고 합시다. 
    이 경우, 그라는 단어 하나에도 '그가', '그에게', '그를', '그와', '그는'과 같이 다양한 조사가 '그'라는 글자 뒤에 띄어쓰기 없이 바로 붙게됩니다.
    위와 같이 단어의 형태가 일정하지 못하고 상황에 따라 뒤섞이고 바뀔 수가 있으므로,
    한국어 토큰화에서는 형태소(morpheme) 란 개념을 반드시 이해해야 합니다. 
    형태소(morpheme)란 뜻을 가진 가장 작은 말의 단위를 말합니다. 
    이 형태소에는 두 가지 형태소가 있는데 자립 형태소와 의존 형태소입니다.
    
    자립 형태소 : 접사, 어미, 조사와 상관없이 자립하여 사용할 수 있는 형태소. 그 자체로 단어가 된다. 
        체언(명사, 대명사, 수사), 수식언(관형사, 부사), 감탄사 등이 있다.
    의존 형태소 : 다른 형태소와 결합하여 사용되는 형태소. 접사, 어미, 조사, 어간을 말한다.
    
    2. 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않습니다.
    앞서 교착어의 특성이 있어서 그런지 한국어는 띄어쓰기가 엄격하지 않아도 문장을 이해하는데 큰 무리가 없습니다.
    고로 언어 자체에서 띄어쓰기에 대한 제약이 적으므로, 사람에 따라 띄어쓰기가 잘 이루어지지 않을 수 있습니다.

- 아래 코드는 텍스트 데이터를 토큰화 하는 기본적인 방식을 몇가지 구현한 코드입니다.
"""

review_text = " 현실과 구분 불가능한 cg. 시각적 즐거음은 최고! 더불어 ost는 더더욱 최고!!"

# (공백으로 나누기)
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
