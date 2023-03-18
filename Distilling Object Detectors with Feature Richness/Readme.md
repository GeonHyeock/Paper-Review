# Distilling Object Detectors with Feature Richness

## Abstract

대부분의 detection기반의 distillation 방법들은 2가지 한계점을 갖는다.
1. bounding box 외부의 유익한 정보를 무시한다. 
2. teacher model로부터 잘못하여 유의미한 객체에 대하여 배경으로 간주한다.

그 이유는 다음과 같다.

<img src="../Distilling Object Detectors with Feature Richness/images/Fig1.jpeg">

1. COCO dataset을 활용하여 모델을 개선한다고 생각해보자. COCO dataset의 category에는 마네킹은 존재하지 않고 사람은 존재한다. 따라서 위의 사진중 밑에 부분을 확인해보면 (a)에 대하여 box정보가 없어 (b)에 아무런 특징이 검출되지 않음을 알 수 있다. \
마네킹과 사람은 비슷하여 유사한 특징을 지니고 있어 마네킹의 특징을 이용해 사람을 detection하는 model을 학습하는데에 도움을 줄 수 있지만 기존의 distillation방법들은 box정보가 없는 data에 대하여 box 외부의 정보를 무시하고 있으므로 마네킹 데이터를 활용하기 어려워 보인다.

2. 위쪽 사진 중 (a)에 대하여 잘못된 예측과 함께 (b)를 보면 확일 할 수 있듯이 사전에 학습된 teacher model로부터 잘못된 예측은 모델 학습에 있어 부정적인 영향을 줄 수 있음을 알 수 있다.

위 2가지 문제점을 FRS(Feature-Richness Score)는 (c)를 보면 알 수 있듯이 1의 문제점 중 특징을 잡아내지 못한 것을 해결하고 2의 문제점 중 잘못된 예측의 box로 인하여 잘못된 훈련을 하는 것을 feature를 줄여주는 것을 통하여 해결하게 된다.

 따라서 FPS는 distillation에서 보다 유용하고 중요한 feature를 찾기 위하여 본 논문에서는 제안한다.

---

## FRS
<img src="../Distilling Object Detectors with Feature Richness/images/Fig2.jpeg">


- $y_c = P(c|f,\theta)$ where c : category, f : feature, $\theta$ : 모델의 파라미터
- $S_l = \max\limits_{c}y_{l,c}$ : 위에서 정의된 확률 $y_c$에 대하여 FPN의 $l$번째 level에 대하여 가장 큰 카테고리의 확률

teacher model로부터 생성된 S의 값으로 FRS Mask를 정의한다.

---

### Distillation Feature Loss

$L_{FPN} = \sum\limits_{l=1}^M\frac{1}{N_l}\sum\limits_{i=1}^W\sum\limits_{j=1}^HS_{l,i,j}\sum\limits_{c=1}^C(F^t_{l,i,j,c} -\phi(F^s_{l,i,j,c}))^2$

- $F^t,F^s$ : teacher와 student model로부터의 feature를 의미한다.

- $\phi : (F^t,F^s)$ 두 feature를 동일한 차원으로 맞추어 주는 conv 연산이다.

- $M$ : FPN의 layer, $W$ : width, $H$ : height, $C$ : category를 의미한다.

즉 FPN으로부터의 $l$번재 특징에 대하여 teacher와 student의 특징값을 비교하는데 한 픽셀에 대하여 특정 object일 가능성이 뚜렷하다면 S의 값이 커져 teacher와 student의 차이값을 더 크게 줄여야 하는 것을 Loss에 반영하였다.

---

### Distillation Head Loss

$L_{head} = \sum\limits_{l=1}^M\frac{1}{N_l}\sum\limits_{i=1}^W\sum\limits_{j=1}^HS_{l,i,j}\sum\limits_{c=1}^CLoss(y_{l,i,j,c}^s,y_{l,i,j,c}^t)$

- Loss : Binary Cross Entropy를 사용하였다.

$L_{FPN}$과 비슷하게 S를 통하여 가중치를 주었고 cls head를 통과한 픽셀별 score를 통하여 bounding box 외부의 정보도 반영을 하고 box 내부의 필요없는 정보를 줄여주는 것을 반영하였다.

---

### Overall Loss

$L = L_{GT} + \alpha L_{FPN} + \beta L_{head}$

위에서 구한 2가지 Loss에 Ground Truth에 대한    Detection loss를 합한것을 최종 Loss로 사용하여 student모델을 학습한다. 이때 $\alpha, \beta$는 scale을 위한 hyperparameter이다.

추가로 FRS의 장점은 다음과 같다.
1. pixel-wise & Fine-Grained
2. FPN 모듈과 잘 어울린다.
3. 다양한 모델에 plug-and-play가 가능하다.

---

## 결과

<img src="../Distilling Object Detectors with Feature Richness/images/Fig3.jpeg">

위 사진은 왼쪽부터 FPN layer에서 낮은 level의 FRS를 시각화 한 결과이다. 작은 영역에서 특징을 잡아주고 오른쪽으로 갈 수록 큰 객체에서 특징을 잡아주는 것을 확인할 수 있다.

<img src="../Distilling Object Detectors with Feature Richness/images/Table1.jpeg">

COCO-Dataset에 대하여 다른 모델들과 비교한 표이다.
- Retina Net & GFL : anchor base인 one-stage detector
- FCOS : anchor free인 one-stage detector
- Faster-RCNN : two-stage detector

FRS는 다양한 모델들에 대하여 성능 향상을 보여주고 있음을 알 수 있다.

[Paper Link](https://arxiv.org/abs/2111.00674)




