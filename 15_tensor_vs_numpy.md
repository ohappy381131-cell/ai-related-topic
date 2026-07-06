# 12. PyTorch 텐서와 NumPy 배열 — 개념과 차이

본 글에서는 PyTorch의 핵심 자료구조인 텐서(Tensor)가 무엇인지, 그리고 NumPy 배열과 어떤 공통점과 차이점을 가지는지를 하나의 흐름 속에서 정리합니다. 두 자료구조가 겉으로는 유사해 보이지만 설계 목적이 근본적으로 다르다는 점을 구조적으로 이해할 수 있도록 설명합니다.

---

## 목차

1. 텐서란 무엇인가
2. NumPy 배열과의 공통점
3. 결정적 차이 1: GPU 연산 지원
4. 결정적 차이 2: 자동 미분 (Autograd)
5. 결정적 차이 3: 계산 그래프 추적
6. 상호 변환: 긴밀한 연동
7. 종합 비교

---

## 1. 텐서란 무엇인가

텐서(Tensor)는 PyTorch에서 데이터를 담는 기본 단위입니다. 수학적으로는 스칼라, 벡터, 행렬을 모두 포괄하는 일반화된 개념입니다.

```
스칼라 (0차원 텐서): 3.14
벡터  (1차원 텐서): [1, 2, 3]
행렬  (2차원 텐서): [[1, 2], [3, 4]]
3차원 텐서:         이미지 한 장 (Height × Width × Channel)
4차원 텐서:         이미지 배치 (Batch × Height × Width × Channel)
```

PyTorch에서 텐서는 단순한 데이터 컨테이너가 아닙니다. 딥러닝 학습에 필요한 모든 연산, 즉 GPU 가속, 자동 미분, 역전파를 지원하도록 설계된 핵심 자료구조입니다.

---

## 2. NumPy 배열과의 공통점

PyTorch 텐서와 NumPy 배열은 겉으로 보기에 매우 유사합니다. 둘 다 다차원 배열을 표현하고, 브로드캐스팅(broadcasting), 슬라이싱, 인덱싱 같은 연산을 지원합니다. 문법도 의도적으로 유사하게 설계되었습니다.

```python
import numpy as np
import torch

# NumPy
a = np.array([1.0, 2.0, 3.0])
print(a * 2)  # [2. 4. 6.]

# PyTorch
b = torch.tensor([1.0, 2.0, 3.0])
print(b * 2)  # tensor([2., 4., 6.])
```

이 유사성은 의도적입니다. NumPy에 익숙한 연구자들이 PyTorch로 쉽게 전환할 수 있도록 설계된 것입니다.

---

## 3. 결정적 차이 1: GPU 연산 지원

NumPy 배열은 CPU에서만 동작합니다. 딥러닝에서 수백만 개의 파라미터를 동시에 연산하려면 CPU만으로는 속도가 턱없이 부족합니다.

PyTorch 텐서는 `.to('cuda')` 한 줄로 GPU로 이동해 연산할 수 있습니다. GPU는 수천 개의 코어가 병렬로 행렬 연산을 처리하므로, 딥러닝 학습 속도가 CPU 대비 수십~수백 배 빨라집니다.

```python
# CPU 텐서
x = torch.tensor([1.0, 2.0, 3.0])

# GPU 텐서로 이동
x = x.to('cuda')  # 또는 x.cuda()

# 다시 CPU로
x = x.to('cpu')
```

NumPy는 이 기능이 없습니다. GPU 연산이 필요하면 CuPy 같은 별도 라이브러리를 사용해야 합니다.

---

## 4. 결정적 차이 2: 자동 미분 (Autograd)

이것이 PyTorch 텐서가 NumPy 배열과 근본적으로 다른 이유입니다.

딥러닝 학습의 핵심은 역전파(backpropagation)입니다. 손실 함수를 각 파라미터에 대해 미분해야 파라미터를 업데이트할 수 있습니다. 파라미터가 수백만 개인 모델에서 이 미분을 수동으로 계산하는 것은 불가능합니다.

PyTorch 텐서는 `requires_grad=True`로 설정하면 해당 텐서에 수행된 모든 연산을 기록합니다. `.backward()`를 호출하는 순간 기록된 연산을 역방향으로 따라가며 각 파라미터의 기울기(gradient)를 자동으로 계산합니다. 이것이 PyTorch의 Autograd 시스템입니다.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2  # y = x²

y.backward()  # dy/dx = 2x 자동 계산
print(x.grad)  # tensor(6.)  ← 2 × 3 = 6
```

NumPy는 이 기능이 전혀 없습니다. 연산 결과만 반환할 뿐, 어떤 연산이 수행되었는지 기록하지 않습니다.

---

## 5. 결정적 차이 3: 계산 그래프 추적

PyTorch 텐서는 연산이 수행될 때마다 내부적으로 계산 그래프(computational graph)를 동적으로 구성합니다. 이 그래프가 Autograd의 기반입니다. 순전파 과정에서 그래프가 만들어지고, 역전파 시 이 그래프를 역방향으로 따라가며 기울기를 계산합니다.

NumPy는 계산 그래프 개념 자체가 없습니다. 단순히 연산 결과값만 반환합니다.

---

## 6. 상호 변환: 긴밀한 연동

PyTorch와 NumPy는 데이터를 공유하는 방식으로 긴밀하게 연동됩니다. 변환 시 메모리를 복사하지 않고 같은 메모리를 참조하므로 효율적입니다.

```python
import numpy as np
import torch

# NumPy → PyTorch
arr = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(arr)

# PyTorch → NumPy
tensor = torch.tensor([1.0, 2.0, 3.0])
arr = tensor.numpy()
```

단, GPU에 올라간 텐서는 `.cpu()`로 먼저 CPU로 내린 뒤 변환해야 합니다. 또한 `requires_grad=True`인 텐서는 `.detach()`로 기울기 추적을 끊은 뒤 변환해야 합니다.

```python
# GPU + requires_grad 텐서를 NumPy로 변환
arr = tensor.detach().cpu().numpy()
```

---

## 7. 종합 비교

| | NumPy 배열 | PyTorch 텐서 |
|---|---|---|
| **주요 목적** | 수치 연산, 데이터 처리 | 딥러닝 모델 학습 |
| **GPU 지원** | 없음 | 있음 (.to('cuda')) |
| **자동 미분** | 없음 | 있음 (Autograd) |
| **계산 그래프** | 없음 | 동적 생성 |
| **메모리 공유** | — | NumPy와 공유 가능 |
| **사용 맥락** | 전처리, 분석, 시각화 | 모델 정의, 학습, 추론 |

---

## 핵심 요약

PyTorch 텐서와 NumPy 배열은 다차원 배열이라는 점에서 유사하고 문법도 비슷하지만, 목적이 다릅니다. NumPy는 수치 연산과 데이터 처리를 위한 도구이고, PyTorch 텐서는 딥러닝 학습에 필요한 GPU 연산과 자동 미분을 지원하도록 설계된 도구입니다. 실무에서는 데이터 전처리는 NumPy로, 모델 학습은 PyTorch 텐서로 처리하고 두 사이를 필요에 따라 변환하는 방식이 일반적입니다.
