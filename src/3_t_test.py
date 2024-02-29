import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

# 표준분포에 따른 데이터 생성
man_height = stats.norm.rvs(loc=170, scale=10, size=500, random_state=1)
woman_height = stats.norm.rvs(loc=150, scale=10, size=500, random_state=1)

# 독립 변수 (성별 키 데이터)
X = np.concatenate([man_height, woman_height])
# 종속 변수 (성별 키에 해당하는 개수)
Y = ["man"] * len(man_height) + ["woman"] * len(woman_height)

# 데이터 시각화
df = pd.DataFrame(list(zip(X, Y)), columns=["X", "Y"])
fig = sns.displot(data=df, x="X", hue="Y", kind="kde")
fig.set_axis_labels("cm", "count")
plt.show()

# t 검정
statistic, pvalue = stats.ttest_ind(man_height, woman_height, equal_var=True)

print("statistic:", statistic)
print("pvalue :", pvalue)
print("*:", pvalue < 0.05)
print("**:", pvalue < 0.001)
