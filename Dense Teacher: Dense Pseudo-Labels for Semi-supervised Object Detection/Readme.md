# Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection

## Abstract
pseudo-boxes에 비하여 Dense Pseudo-Label은 후처리를 필요로 하지 않으므로 더 많은 정보를 포함하고 있다.

<img src="../Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection/images/Fig1.jpeg">

동일한 이미지에 대하여 각기 다른 augmentation과 dropout 등을 적용하여 teacher model로 부터 나온 output을 통하여 student 모델을 학습하는 기법은 좋은 성능을 보여주고 있다. 그러나 전통적인 SSOD(Semi Supervised Object Detection)의 task의 경우 pseudo-boxes를 만드는 과정에서 후처리를 위한 NMS threshold 인 $\sigma_{nms}$와 score threshold인 $\sigma_{t}$와 같은 hyper-parameter에 모델의 성능이 의존하게 된다. 

본 논문에서는 Dense-Teacher라는 새로운 pipline을 제안한다.

---

## DPL : Dense Pseudo Label

DPL은 integral label이다. 일반적으로 사람이 읽을 수 있는 형태의 label이 아닌 후처리 없이 network로 부터 나오는 label이다.

student 모델의 EMA를 통하여 생성된 teacher model로부터 unlabeled data에 대하여 DPL을 생성한다. 이후 student model을 labeled data와 unlabeled data로 부터 학습한다.

위 그림에서 볼 수 있듯이 DPL은 사후처리를 필요로 하지 않으므로 Dense Teacher의 pipe line은 단순하다.

---

## Dense Teacher

먼저 SSOD(Semi Supervised Object Detection)의 framework에 대하여 설명하겠다.

### Pseudo-Labeling Framework
1. labeled image와 unlabeled image를 무작위로 샘플링 한다.
2. student 모델의 weight를 EMA방법을 통하여 생성된 teacher 모델로 부터 pseudo-label을 생성한다.
3. labeled image를 통하여 student model을 훈련시키고 supervised loss $L_s$를 구한다. unlabeled image와 함께 pseudo-label를 사용하여 unsupervised loss $L_u$를 구한다.
4. 위에서 구한 2가지 loss를 통하여 student model을 학습하고 EMA방법을 사용하여 teacher model의 weight를 업데이트 한다.

최종적인 Loss는 다음과 같이 정의된다. \
$L = L_u + w_u L_u$ ,이때 $w_t$는 unsupervised loss weight이다. \

전형적으로 $L_u$는 pseudo-box들로부터 생성된다. 하지만 본 논문에서는 pseudo-box로부터 $L_u$를 구하는 방법이 최선의 선택이 아니라고 주장한다.

---

### Disadvantages of pseudo-box label


본 논문에서는 pseudo-box가 최선의 선택이 아니라고 주장한다. 그 이유는 다음과 같다.

1. Dilemma in Threshold and NMS 
<img src="../Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection/images/Fig2.jpeg">

    (a)는 teacher model의 output에 대한 threshold값의 변화에 따른 성능을 나타낸 것이다. 이때 회색 배경의 X표시는 모델 훈련 과정에서 수렴하지 않았음을 의미한다. 이러한 결과는 (c),(d)를 보았을때 teacher model로 부터의 예측값에 False Nagative 값이 많았을 것이라고 예상된다. - (1)
    
    (b)에서 볼 수 있듯이 후처리 과정에서 필요한 NMS의 임계값에 따라 모델의 성능이 변화한다. - (2)

    (1),(2)에 따라 hyperparameter를 tuning하는 것은 추가적인 작업이 필요할 뿐만 아니라 완벽한 값을 찾는 것은 어려운 일이다. 이처럼 pseudo-box를 생성하기 위한 hyperparameter를 결정하는 것 자체가 pseudo-box label의 단점이 된다.

2. Inconsistent Label 
    <img src="../Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection/images/Fig3.jpeg">
    
    pseudo-box를 결정하기 위하여 사전에 정의된 비율에 따라 anchor box를 생성하여 model의 output을 예측하는 것은 detection task에서 매우 당연한 일이다. 하지만 위 그림에서 볼 수 있듯이 Ground Truth와 Pseudo Box에 대한 IOU값이 0.5이고 이를 훈련에 사용하게 되면 모델의 성능이 떨어질 것으로 보여진다. 따라서 본 논문에서는 pseudo-box를 label로 사용하는 방법이 좋지 못하는 방법이라고 주장한다.

---

## Dense Pseudo-Label

위에서 주장하는 pseudo-box의 단점을 극복하고자 본 논문에서는 Dense Pseudo-Label을 이야기한다.



DPL은 post-sigmoid logit으로 부터 나온 예측치를 사용한다. 따라서 연속적인 값으로 0~1 사이의 값을 갖는다. Loss 함수로 Quality Focal Loss를 사용하였다.

threshold로 예측치를 제거하는 연산이 없어 낮은 점수의 예측값도 보존이 된다. 낮은 점수의 예측값은 대부분 배경을 포함하므로 좋은 정보라 할 수 없다. 따라서 [FRS](https://arxiv.org/abs/2111.00674)기반으로 학습영역과 억제영역으로 나눈다.

$$S_i = \max\limits_{c \in [1,C]}(p_{i,t}^t)$$

```math
y_i=
\begin{cases}
p_i^t, if S_i\ in\ top\ k\% \\
0, otherwise
\end{cases}
```

이때 $p_{i,c}^t$는 teacher model로부터 나온 c번째 class의 i번째 샘플의 score 예측치를 이고 $C$는 클래스의 갯수이다.

$L^{cls}_i = -|y_i - p^s_i|^r * [y_ilog(p^s_i) + (1-y_i)log(1-p^s_i)]$ 

where  $y_i$ : teacher prediction(DPL), $p^s_i$ : studetn prediction, $r$ : Loss의 스케일 상수

<img src="../Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection/images/Table1.jpeg">
Our Divison의 의미는 FRS를 이용하여 나눈 score를 의미한다.
COCO-Standard 10%에 대한결과이다. *표시가 되어있는 것은 FCOS에서 구현한 것을 COCO데이터에 대하여 재구현한 결과이다.

위의 표를 통하여 FRS score를 사용한 것이 사용안한 것 보다 좋은 성능을 보여주는 것을 알 수 있다.


---

<img src="../Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection/images/Table2.jpeg">
Dense Teacher와 다른 방법론들에 대하여 COCO-Standard데이터와 비교한 표이다.

[Paper Link](https://arxiv.org/abs/2207.02541)