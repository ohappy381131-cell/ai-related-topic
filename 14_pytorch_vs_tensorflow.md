# PyTorch와 TensorFlow — 딥러닝 프레임워크 비교

본 글에서는 딥러닝의 양대 프레임워크인 PyTorch와 TensorFlow를 등장 배경부터 핵심 차이, 개발 경험, 생태계, 현재 트렌드까지 하나의 흐름 속에서 비교합니다. 두 프레임워크가 각각 어떤 철학으로 설계되었고, 어떤 상황에서 어떤 것을 선택하는 것이 적합한지를 구조적으로 이해할 수 있도록 설명합니다.

---

## 목차

1. 두 프레임워크의 등장 배경
2. 핵심 차이: 계산 그래프 방식
3. 문법과 개발 경험
4. 연구 vs 프로덕션
5. PyTorch Lightning: 중간 어딘가가 아니라 PyTorch 위의 도구
6. 배포 관점에서의 비교
7. 생태계와 커뮤니티
8. 현재 트렌드

---

## 1. 두 프레임워크의 등장 배경

딥러닝 모델을 처음부터 직접 구현하려면 행렬 연산, 역전파, GPU 연산 등 방대한 저수준 코드가 필요합니다. 딥러닝 프레임워크는 이 복잡한 작업을 추상화해 연구자와 개발자가 모델 설계에 집중할 수 있도록 돕는 도구입니다.

**TensorFlow**는 2015년 Google이 공개했습니다. 당시 구글 내부에서 사용하던 머신러닝 시스템 DistBelief를 발전시킨 것으로, 처음부터 대규모 프로덕션 배포를 염두에 두고 설계되었습니다. 초기에는 정적 계산 그래프(static computational graph) 방식을 채택해, 모델을 먼저 완전히 정의한 뒤 실행하는 구조였습니다.

**PyTorch**는 2017년 Meta(당시 Facebook) AI Research가 공개했습니다. 연구자들이 실험을 빠르게 반복할 수 있도록 동적 계산 그래프(dynamic computational graph) 방식을 채택했습니다. Python과 자연스럽게 통합되는 설계 철학이 연구 커뮤니티에서 빠르게 호응을 얻었습니다.

---

## 2. 핵심 차이: 계산 그래프 방식

두 프레임워크의 가장 근본적인 차이는 계산 그래프를 어떻게 다루는가입니다. 이 차이가 개발 경험 전반에 영향을 미칩니다.

TensorFlow 초기 버전은 **Define-and-Run** 방식이었습니다. 코드를 실행하기 전에 모델의 전체 계산 그래프를 먼저 정의(define)하고, 그다음에 데이터를 흘려보내(run) 실행합니다. 그래프가 고정되어 있으므로 최적화와 배포에 유리하지만, 디버깅이 어렵고 직관적이지 않다는 단점이 있었습니다. 오류가 발생해도 어디서 문제가 생겼는지 추적하기 어려웠습니다.

PyTorch는 **Define-by-Run** 방식입니다. 코드가 실행되는 순간 계산 그래프가 동적으로 생성됩니다. Python 코드가 실행되는 흐름 그대로 그래프가 만들어지므로, 일반 Python 디버거로 중간 값을 확인하고 조건문이나 반복문을 자유롭게 사용할 수 있습니다. 실험하고 수정하는 연구 환경에 훨씬 적합합니다.

TensorFlow는 2.0 버전(2019년)부터 Eager Execution을 기본으로 채택해 PyTorch와 유사한 동적 방식을 지원하기 시작했습니다. 이 변화로 두 프레임워크의 개발 경험 차이가 많이 줄어들었습니다.

---

## 3. 문법과 개발 경험

PyTorch는 Python스럽습니다. NumPy와 매우 유사한 문법을 가지고 있어, NumPy에 익숙한 사람이라면 진입 장벽이 낮습니다. 코드가 직관적이고 읽기 쉬우며, 일반 Python 디버거(pdb, VS Code 디버거 등)를 그대로 사용할 수 있습니다. 무엇보다 학습 루프를 처음부터 직접 작성하므로 에포크마다 무슨 일이 일어나는지, 손실을 어떻게 계산하는지, 파라미터를 언제 업데이트하는지를 코드로 전부 제어할 수 있습니다. 실험 중간에 학습률을 바꾸거나 특정 조건에서 다른 손실 함수를 적용하는 것이 자연스럽습니다.

```python
# PyTorch: 직관적인 Python 스타일
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
output = model(torch.randn(32, 784))
```

TensorFlow는 Keras API를 통해 높은 수준의 추상화를 제공합니다. `model.fit()` 한 줄로 학습을 처리할 수 있어 간단한 모델은 매우 빠르게 구현됩니다. 간결하지만 내부 흐름을 직접 제어하기 위해서는 커스텀 학습 루프를 별도로 작성해야 합니다.

```python
# TensorFlow + Keras: 간결한 고수준 API
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,))
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=5)
```

두 프레임워크 모두 고수준 API와 저수준 커스터마이징을 지원합니다. 다만 PyTorch는 저수준 제어가 기본값이고, TensorFlow는 고수준 API(Keras)가 기본값입니다.

---

## 4. 연구 vs 프로덕션

두 프레임워크가 역사적으로 강점을 보여온 영역이 다릅니다.

PyTorch는 연구 커뮤니티에서 압도적입니다. arXiv에 올라오는 논문의 대부분이 PyTorch 기반이며, Hugging Face의 Transformers 라이브러리, Meta의 LLaMA 등 최신 모델 대부분이 PyTorch로 구현됩니다. 동적 그래프 방식이 주는 유연성 덕분에 빠른 실험과 프로토타이핑에 강합니다.

TensorFlow는 프로덕션 배포에 강점이 있습니다. TensorFlow Serving, TensorFlow Lite(모바일), TensorFlow.js(브라우저) 등 다양한 배포 환경을 위한 도구가 잘 갖춰져 있습니다. Google의 인프라와 긴밀하게 통합되어 있어 대규모 서비스 환경에서 검증된 신뢰성을 가집니다.

---

## 5. PyTorch Lightning: 중간 어딘가가 아니라 PyTorch 위의 도구

PyTorch Lightning은 PyTorch와 TensorFlow의 중간이 아닙니다. **PyTorch 위에서 동작하는 래퍼(wrapper) 라이브러리**입니다. PyTorch의 자유도는 그대로 유지하면서, 반복적으로 작성해야 하는 boilerplate 코드(학습 루프, 로깅, 체크포인트 저장 등)를 자동화해주는 도구입니다.

```
PyTorch          : 자유도 높음, 코드 많음
PyTorch Lightning: 자유도 유지, 코드 간결
TensorFlow/Keras : 자유도 낮음, 코드 간결
```

즉 PyTorch Lightning은 TensorFlow에 가까워지는 것이 아니라, **PyTorch의 철학을 유지하면서 생산성을 높이는** 도구입니다. 연구자가 PyTorch의 유연함을 포기하지 않으면서도 코드를 깔끔하게 관리할 수 있게 해줍니다.

---

## 6. 배포 관점에서의 비교

배포 환경에서는 실험의 유연성보다 **안정성과 재현성**이 중요합니다. 매번 다르게 돌아가는 동적 코드보다, 한 번 정의된 구조가 고정되어 예측 가능하게 작동하는 것이 필요합니다.

PyTorch는 동적 그래프라 배포 시 그래프를 고정하는 추가 작업(TorchScript 변환, ONNX 내보내기)이 필요합니다. PyTorch Lightning은 이 배포 과정을 표준화해서 더 쉽게 만들어줍니다. TensorFlow는 원래 정적 그래프 기반으로 설계되었기 때문에 배포 파이프라인이 더 자연스럽게 연결됩니다.

```
연구 (유연성 중심)              배포 (안정성 중심)

PyTorch ──────────────────────── TorchScript/ONNX
                                        ↑
PyTorch Lightning ───────────────────── ┤
                                        ↑
TensorFlow/Keras ─────────────── TF Serving
```

배포 관점에서만 보면 PyTorch → PyTorch Lightning → TensorFlow 순으로 배포 친화적인 방향으로 이동하는 흐름으로 읽힙니다. 단, 이것은 배포라는 하나의 축에서 성립하는 이야기이고, PyTorch Lightning이 TensorFlow에 가까워지는 것이 목적은 아닙니다.

---

## 7. 생태계와 커뮤니티

| | PyTorch | TensorFlow |
|---|---|---|
| **주요 후원** | Meta | Google |
| **연구 커뮤니티** | 압도적 우세 | 상대적으로 적음 |
| **논문 구현** | 대부분 PyTorch | 일부 TensorFlow |
| **배포 도구** | TorchServe, TorchScript | TF Serving, TF Lite, TF.js |
| **모바일/엣지** | 상대적으로 약함 | 강함 |
| **학습 곡선** | 완만 (Python 친화적) | 초기 가파름, Keras 이후 완만 |
| **대표 라이브러리** | Hugging Face, Lightning | Keras, TF-Agents |

---

## 8. 현재 트렌드

2020년을 기점으로 PyTorch가 연구 분야에서 TensorFlow를 압도하기 시작했습니다. 주요 AI 연구소(OpenAI, Meta, Hugging Face 등)가 PyTorch를 기본 프레임워크로 사용하고, 대형 언어 모델 시대가 열리면서 이 흐름이 더 강해졌습니다. 현재 AI 연구와 개발의 주류는 PyTorch입니다.

TensorFlow는 Google 내부와 기존에 TensorFlow 기반으로 구축된 대규모 프로덕션 시스템에서 여전히 강세를 보입니다. 또한 Google이 JAX라는 새로운 프레임워크를 내부적으로 밀고 있어, TensorFlow의 위상이 장기적으로 어떻게 변할지는 지켜봐야 합니다.

---

## 핵심 요약

| | PyTorch | TensorFlow |
|---|---|---|
| **설계 철학** | 연구자 친화적, 유연성 | 프로덕션 친화적, 안정성 |
| **그래프 방식** | 동적 (Define-by-Run) | 정적→동적 혼합 (2.0 이후) |
| **강점** | 연구, 실험, 최신 모델 | 배포, 모바일, 대규모 서비스 |
| **현재 트렌드** | 연구·산업 모두 주류 | 프로덕션·Google 생태계 |

처음 딥러닝을 시작하거나 최신 연구를 따라가려면 PyTorch가 현실적인 선택입니다. 이미 TensorFlow 기반 인프라가 구축된 환경이거나 모바일·엣지 배포가 핵심이라면 TensorFlow가 여전히 유효합니다.
