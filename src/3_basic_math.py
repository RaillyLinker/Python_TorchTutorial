import matplotlib.pyplot as plt
import numpy as np

"""
[딥러닝 기본 수학]
- 여기선 딥러닝에 사용되는 가장 기본적인 수학에 대해 정리 하겠습니다.
    수학에 약한 저라고 하더라도 이해가 될 정도로 그렇게 어렵지는 않은 내용으로,
    되도록 쉽고 자세히 설명할 것이니 두려워할 것은 없습니다.
"""

"""
(일차함수)
- 일차함수의 가장 기본적인 형태는,
    y = ax + b
    로 볼 수 있습니다.
    입력값 x 가 있을 때, 이에 a 를 곱하고 b 를 더한 것으로,
    a 는 기울기라고 하고, b 는 절편이라고 합니다.
    일차 함수의 일차라는 것은 위 수식에서 a 의 지수를 의미합니다.
    지수란, a^2, a^3 와 같이 제곱, 세제곱을 나타내는 것에서, a^2 에서는 2, a^3 에서는 3 과 같이 위에 달리는 숫자로,
    아시다시피 a 를 몇번 곱하느냐에 대한 값인데, 함수 내에 가장 지수가 큰 기울기의 지수가 1 이라면 일차함수, 2 라면 이차함수라고 합니다.
    
- 일차함수는 그래프로 나타내면 선이 하나 그어지는 것으로,
    기울기인 a 의 경우는, x 가 증가 할 때 y 가 변화하는 정도로, 예를들어 기울기가 1 이라면 x 의 증가값과 동일하게 증가하고,
    2 라면 y 는 x 의 2배씩 증가하게 될 것이며, 0이라면 y 는 x 값에 전혀 영향을 받지 못하게 됩니다.
    절편인 b 의 경우는 x 가 0 일 때의 y 의 초기값으로 생각하면 됩니다.
"""
# x 값 생성 (-10부터 10까지 0.1 간격으로)
x = np.arange(-10, 10, 0.1)
# 일차함수 식
y = 2 * x + 3
# 그래프 그리기
plt.plot(x, y, label=f'2x + 3')
plt.title('Linear Function')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# (이차함수의 그래프)
# 이차함수 그래프를 본다면 아래(기울기가 0보다 클 때)로 불룩한 토기 모양입니다.
# x 값 생성 (-10부터 10까지 0.1 간격으로)
x = np.arange(-10, 10, 0.1)
# 일차함수 식
y = x ** 2
# 그래프 그리기
plt.plot(x, y, label=f'x ** 2')
plt.title('Quadratic Function')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

"""
(미분)
- 위와 같은 이차 함수의 그래프는 곡선을 그리고 있습니다.
    x 가 1에서 2로 갈 때의 y의 변화 정도와, x 가 4에서 5로 갈 때의 y의 변화 정도는 다를 것입니다.
    그 간격을 더 작게 하여, 1에서 1.5 로 갈 때, 1 에서 1.00000001 로 갈 때의 y 값의 변화 정도는 또 다를 것입니다.
    이렇게 작게 쪼갠다고 하여 미분입니다.
    이러한 방식으로 x라는 점에서의 기울기를 알아낼 수 있게 됩니다.
    
- 미분 기울기를 구하는 공식
    (y 값의 증가량 / x 값의 증가량) = ((f(b) - f(a)) / (b - a)) = ((f(a + dx) - f(a)) / (dx)) = (dy / dx)
    위 공식에서 dx 는 x 값의 극소 증가값입니다. 
    dy 는 dx 가 적용 되었을 때의 y 의 증가 값이죠.
    dx 로 dy 를 나눈다는 것은, dx 의 값이 dy 에 끼치는 영향력을 나타낸 것입니다.
    예를 들어 봅시다. x 가 1 일때의 기울기를 구하고 싶다면,
    함수에 넣을 수 있는 매우 작은 값을 0.0001 이라고 결정했다고 합시다.
    그러면 x 가 1.0001 일때의 y 값과 x 가 1 일때의 y 값의 차이를 구하면 dy 가 준비되겠죠?
    dx 의 경우는 1.0001 에서 1 을 빼면 됩니다.
    
- 미분 법칙
    1. 함수 f(x) = x 일 때, f(x) 의 미분 함수인 f`(x) = 1 입니다. (함수에서 입력값과 동일한 값이 반환되니 기울기는 1)
    2. f(x) = a 에서 a 가 상수일 때, f`(x) = 0 (함수의 입력값이 어떻든 동일한 상수값이 반환되므로 기울기가 없습니다.)
    3. f(x) = ax 에서 a 가 상수일 때, f`(x) = a (입력값에 a 가 곱해져 나오는 형태이므로, a 가 즉 기울기가 됩니다.)
    4. f(x) = x^a 에서 a 가 자연수일 경우, f`(x) = ax^(a-1)
    5. 함성 함수 f(g(x)) 에서 f(x) 와 g(x) 가 미분 가능할 때 {f(g(x))}` = f`(g(x)) * g`(x) (합성 함수의 기울기는 각 함수의 미분값의 곱과 같습니다.)
    
- 편미분
    함수 내에 변수가 여러개 있을 때, 특정 변수만의 기울기를 구하는 것,
    나머지 변수는 특정 값으로 고정하여 상수로 보고 미분을 하는 것을 편미분이라 부릅니다.
    예를들어 f(x1, x2) = x1^2 + (x1 * x2) + a 라는 함수가 있을 때, x1 에 대해 편미분을 한다고 하면,
    df / dx1 이라고 표시를 하며, 미분 법칙에 따라 정리하면,
    df / dx1 = 2x1 + x2 가 됩니다.
    4번 법칙으로 2x1 이 나온 것이고, 2번 법칙에 대응했을 때, 
    x2 가 편미분으로 상수취급이 되므로 x2 가 나온 것이고, 
    a 의 경우는 상수이므로 2번 법칙으로 0 이 된 것입니다.
"""
# (미분 결과 그래프)
# 특정 미분 위치에서 미분을 수행한 결과를 그래프로 나타냅니다.
# 미분 위치
differentiation_x = 7


# 이차 함수 그래프
def quadratic_function(x):
    return x ** 2


# 미분 함수
def derivative_quadratic(x):
    return 2 * x


# x 값 생성 (-10부터 10까지 0.1 간격으로)
x_positive = np.arange(0, 10, 0.1)
# 이차함수 값 계산 (음수 부분 제외)
y_positive = quadratic_function(x_positive)
# 미분값 계산 (x=7에서의 미분값)
derivative_at_x7 = derivative_quadratic(differentiation_x)
# 미분값을 이용하여 접선 계산
tangent_line_at_x7_positive = derivative_at_x7 * (x_positive - differentiation_x) + quadratic_function(
    differentiation_x)
# 그래프 그리기
plt.plot(x_positive, y_positive, label=f'x ** 2 (positive)')
plt.plot(x_positive, tangent_line_at_x7_positive, label=f'Tangent at x=7', linestyle='--', color='red')
plt.scatter(differentiation_x, quadratic_function(differentiation_x), color='red')  # 해당 점 표시
plt.title('Quadratic Function and Tangent at x=7 (positive)')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

"""
(지수와 지수함수)
- 지수는 앞서 설명했듯, 숫자의 제곱 횟수입니다.
    1제곱이라면 지수는 1, 제곱이라면 2, 세제곱이라면 3 입니다.
    
- 지수의 아래에 있는 제곱되는 대상의 숫자는 밑이라고 부릅니다.

- 지수함수란, 변수 x 가 지수 자리에 있는 함수를 의미합니다.
    y = a^x
    와 같습니다.
    이러한 지수함수에서는 밑의 값이 무엇인지가 중요한데, 이 값이 1 이라면 함수가 아니며, 0 보다 작으면 허수를 포함하게 되므로 안됩니다.
    
- 아래는 밑의 값이 1 보다 클 경우와 0보다 크고 1 보다 작을 때의 경우를 나타낸 지수함수의 그래프입니다.
"""
# x 값 생성
x = np.linspace(-2, 2, 400)
# a > 1 일 때의 지수함수 그래프 (a=2)
y1 = 2 ** x
# 0 < a < 1 일 때의 지수함수 그래프 (a=0.5)
y2 = 0.5 ** x
# 그래프 그리기
plt.plot(x, y1, label=r'$y=2^x$ (a > 1)')
plt.plot(x, y2, label=r'$y=0.5^x$ (0 < a < 1)')
# 그래프 스타일 및 레이블 설정
plt.title('Exponential Functions with Different Values of a')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
