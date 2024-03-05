import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

"""
[T-Test 코드]
아래 코드는 기본적인 통계 기법인 T-Test 를 넘파이로 구현하여 확인한 것입니다.
독립변수인 신장 데이터가 종속 변수인 성별 라벨 데이터에 유의한 영향을 끼치는지를 확인합니다.
데이터 선정 단계에서 사용하는 기법의 일종입니다.
"""

# 표준분포에 따른 데이터 생성
man_height = stats.norm.rvs(loc=170, scale=10, size=500, random_state=1)
woman_height = stats.norm.rvs(loc=150, scale=10, size=500, random_state=1)

# 독립 변수 (성별 키 데이터)
X = np.concatenate([man_height, woman_height])
# 5 개 출력 : [186.24345364 163.88243586 164.71828248 159.27031378 178.65407629]
print(X[:5])

# 종속 변수 (성별 키에 해당하는 개수)
Y = ["man"] * len(man_height) + ["woman"] * len(woman_height)
# 5 개 출력 : ['man', 'man', 'man', 'man', 'man']
print(Y[:5])

# t 검정
statistic, pvalue = stats.ttest_ind(man_height, woman_height, equal_var=True)

print("statistic:", statistic)  # 출력 : 31.96162891312776
print("pvalue :", pvalue)  # 출력 : 6.2285854381989205e-155
print("*:", pvalue < 0.05)  # 출력 : True (유의함)
print("**:", pvalue < 0.001)  # 출력 : True (매우 유의함)

# X, Y 데이터 시각화
df = pd.DataFrame(list(zip(X, Y)), columns=["X", "Y"])
fig = sns.displot(data=df, x="X", hue="Y", kind="kde")
fig.set_axis_labels("cm", "count")
plt.show()
