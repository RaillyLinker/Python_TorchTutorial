import re, collections

"""
[하위 단어 토크나이징]
인간이 이해하는 방식으로 자연어 데이터 분석을 하기 위해서는 공백 단위 텍스트 분리보다는 형태소 단위의 텍스트 분리가 효과적일 것입니다.
하지만 언어라는 것은 형태소일지라도 시간이 지남에 따라 생성, 소멸, 수정의 변화가 있을 수 있습니다.
또한 휴먼 에러로 인하여 오탈자로 인한 오류가 있을 수도 있죠.
즉, 단순한 형태소 분석은 신조어, 외래어, 전문용어, 고유어 등을 처리할 때 약점을 보입니다.
이렇게 정적인 형태소 분리에서 만들어진 단어 사전(Vocabulary) 에 없는 단어가 입력 되었을 때를 OOV(Out Of Vocabulary), 혹은 UNK(Unknown Token) 라고 부릅니다.
자연어 처리에서는 이러한 OOV 를 고려해야만 하는 것입니다.
이를 해결하기 위한 방법 중 하나는 분리된 단어를 더 작은 단위로 쪼개는 방법이 있습니다.
예를들어 영어의 birthplace 는 birth + place 로 나눌 수 있고, 그렇게 나누어진 단어들의 조합으로 실제 의미를 알 수 있습니다.
한국어로 하자면 수영장은 수영 + 장, 안경태의 경우는 안경 + 태...
이와같이 단어의 합으로 이루어진 단어가 많이 있으며, 사전에는 없는 신조어가 입력되더라도 이러한 작업으로 어느정도 그 의미를 유추할 수 있는 것이죠.
이것을 하위 단어 토큰화라고 합니다.
하위 단어 토큰화에는 두가지 방식이 존재합니다.
"""

"""
[바이트페어 인코딩]
바이트페어 인코딩 (Byte Pair Encoding : BPE) 란, 다이어그램 코딩(Diagram Coding) 이라고도 합니다.
1994년에 제안된 데이터 압축 알고리즘으로 시작된 방식이지만, 지금은 이처럼 자연어 처리(NLP) 분야, 
특히 텍스트 또는 언어 모델링에서 토큰화(tokenization) 기법으로 널리 사용되고 있습니다.

이 방식의 기본 아이디어는 말뭉치(corpus) 내에서 가장 빈번하게 등장하는 바이트 쌍(혹은 문자 쌍)을 반복적으로 합치면서 새로운 단어 또는 토큰을 생성하는 것입니다.
이렇게 함으로써, 모델이 처리해야 할 토큰의 수를 줄이고, 언어의 다양한 변형(variation)을 더 잘 핸들링할 수 있게 됩니다.

예시로 설명하겠습니다.

아래와 같은 문자열이 주어졌을 때 BPE을 수행한다고 해봅시다.

aaabdaaabac

BPE은 기본적으로 연속적으로 가장 많이 등장한 글자의 쌍을 찾아서 하나의 글자로 병합하는 방식을 수행합니다. 
태생이 압축 알고리즘인 만큼, 여기서는 글자 대신 바이트(byte)라는 표현을 사용하겠습니다. 
예를 들어 위의 문자열 중 가장 자주 등장하고 있는 바이트의 쌍(byte pair)은 'aa'입니다. 
이 'aa'라는 바이트의 쌍을 하나의 바이트인 'Z'로 치환해보겠습니다.

ZabdZabac
Z=aa

위와 같이 두개씩 가장 많이 나타나는(그리고 가장 처음 발견된) aa 를 Z 로 치환하고, 이 Z 에 aa 가 할당되어 있다는 것을 기록해둡니다.
그리고 다음으로 위 문자열 중에서 가장 많이 등장하고 있는 바이트의 쌍은 'ab'입니다. 이 'ab'를 'Y'로 치환해봅시다.

ZYdZYac
Y=ab
Z=aa

자, 다시 해봅시다. 치환된 글자 역시 압축 대상에서 빠질 수가 없습니다.
다음으로 가장 많이 등장하고 있는 바이트의 쌍은 'ZY'입니다. 이를 'X'로 치환해봅시다.

XdXac
X=ZY
Y=ab
Z=aa

이제 더 이상 병합할 바이트의 쌍은 없으므로 BPE는 위의 결과를 최종 결과로 하여 종료됩니다.

위와 같이 문자를 치환하는 과정에서 단어 사전을 만들 수 있으므로 자연어 처리에 사용 할 수 있는 것입니다.

BPE을 요약하면, 글자(charcter) 단위에서 점차적으로 단어 집합(vocabulary)을 만들어 내는 Bottom up 방식의 접근을 사용합니다.
고로 실제로 의미를 지닌 단어를 분리한다고 단언할 수는 없지만, 많은 데이터에서 이처럼 단어를 뽑아내다보면 확률상 발생 빈도가 높은 글자들의 조합이
생겨나게 될 것이므로 신뢰성이 있는 방식입니다.

이러한 방식을 통해 모델은 미처 학습하지 못한 단어들도 서브워드 또는 문자의 조합으로 표현하고 이해할 수 있게 됩니다.
예를 들어, 'un-'와 같은 접두사나 '-ing'과 같은 접미사를 포함한 단어들은 학습 과정에서 직접 마주치지 않았더라도,
이들의 구성 요소를 통해 의미를 유추할 수 있습니다.
BPE는 따라서 언어 모델이 새로운 단어나 희귀 단어에 대해 더 나은 일반화(generalization) 능력을 가지도록 돕습니다.

(BPE 의 구현)
먼저, 말뭉치에서 단어를 추려옵니다.

# dictionary
l o w : 5,  l o w e r : 2,  n e w e s t : 6,  w i d e s t : 3

위와 같이 말뭉치 안에서 찾아낸 단어와, 해당 단어가 몇번 사용되었는지에 대한 카운트를 준비해둡니다.
low 가 5 번, lower 가 2번... 이렇게 준비하고, 글자 단위로 나누어(바이트 단위로) 준비하세요.

이때는 글자 단위로 나뉜 것이므로 중첩되지 않는 글자들을 모두 사전으로 등록하면 됩니다.

l, o, w, e, r, n, s, t, i, d

이렇게 초기 사전이 형성 되네요.

자, 이제부터는 위에서 간단히 설명한대로 탐색 및 치환을 반복하면 됩니다.

먼저 위 데이터에서 탐색을 해봅시다.

2개씩 탐색을 하니, e s 가 빈도가 가장 많네요.
newest 가 말뭉치에서 6번, widest 가 3 번 탐색되었으니 합쳐서 9번 발생한 것이 됩니다.

그러면 이를 치환합시다.

이때, 압축을 위한 것이 아니라 하위 단어를 선정하기 위한 것이므로,
e, s 가 아니라 es 로 합쳐서

l o w : 5,
l o w e r : 2,
n e w es t : 6,
w i d es t : 3

이렇게 됩니다.
이제 기존 사전에서 es 가 합쳐져서,

l, o, w, e, r, n, s, t, i, d, es

사전은 위와 같이 될 것입니다.

자, 한번 탐색 및 치환을 했습니다.
이제 다음 탐색을 해봅시다.

다음에는 치환된 es 와 t 가 쌍으로 하여 빈도가 9번으로 가장 많네요.

l o w : 5,
l o w e r : 2,
n e w est : 6,
w i d est : 3

l, o, w, e, r, n, s, t, i, d, es, est

위와 같이 치환됩니다.

다음으로 빈도가 많은 쌍은 또 뭘까요?

l 과 o 입니다.

lo w : 5,
lo w e r : 2,
n e w est : 6,
w i d est : 3

l, o, w, e, r, n, s, t, i, d, es, est, lo

이러합니다.

이렇게 반복하여 10번을 실행하였다고 한다면,

low : 5,
low e r : 2,
newest : 6,
widest : 3

l, o, w, e, r, n, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest

이렇게 되네요.

자, 보다시피 글자에서 시작하여 단어가 합성되어가며 단어 사전을 만들 수 있음을 알 수 있습니다.

이와 같이 하여 OOV 를 줄일 수 있습니다.

실제 구현은 아래와 같습니다.
"""

# BPE 구현
# 탐색 -> 치환 반복 횟수
num_merges = 10

# 말뭉치에서 단어 / 빈도 사전 추출하기
dictionary = {
    'l o w': 5,
    'l o w e r': 2,
    'n e w e s t': 6,
    'w i d e s t': 3
}

# 하위 단어 사전
sub_word_vocab = []
bpe_codes = {}
bpe_codes_reverse = {}

# 초기 단어 사전 구성
sub_word_vocab_set = set()
for key in dictionary.keys():
    letters = [char for char in key if char.isalpha()]
    sub_word_vocab_set.update(letters)

sub_word_vocab = list(sub_word_vocab_set)
print('초기 하위 단어 사전 :', sub_word_vocab, '\n')

for i in range(num_merges):
    print(i + 1)

    # 유니그램의 pair들의 빈도수를 카운트
    pairs = collections.defaultdict(int)
    for word, freq in dictionary.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    print('현재 pair들의 빈도수 :', dict(pairs))

    pairs = pairs
    best = max(pairs, key=pairs.get)
    v_out = {}
    bigram = re.escape(' '.join(best))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in dictionary:
        w_out = p.sub(''.join(best), word)
        v_out[w_out] = dictionary[word]
    dictionary = v_out

    sub_word_vocab.append(''.join(best))

    bpe_codes[best] = i
    bpe_codes_reverse[best[0] + best[1]] = best

    print("new merge: {}".format(best))
    print("dictionary: {}".format(dictionary))
    print("sub_word_vocab: {}".format(sub_word_vocab))
    print("bpe_codes: {}".format(bpe_codes))
    print("bpe_codes_reverse: {}".format(bpe_codes_reverse))
    print("")

"""
[워드피스]
Wordpiece 토크나이저는 바이트 페어 인코딩 토크나이저와 듀사한 방법으로 학습되지만,
빈도 기반이 아닌 확률 기반으로 글자 쌍을 병합합니다.

"""

# todo
