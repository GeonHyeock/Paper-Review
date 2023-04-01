# End-to-End Object Detection with Transformers

## Abstract

<img src="../End-to-End Object Detection with Transformers/image/Fig2.jpeg">

본 논문에서는 object detection을 수행하기 위하여 direct set prediction을 제시한다. 이러한 framework를 DEtection TRansformer 즉 DETR이라고 이름을 지었다. DETR은 task의 사전정보를 통하여 설정하는 NMS와 anchor box와 같이 사람이 직접 설계하는 절차를 수행하지 않아도 된다. 또한 train을 위해 set-based의 loss를 이분매칭과 함께 사용하여 모델을 훈련했다.

---

## Loss : object detection set prediction loss


DETR은 고정된 N개의 output을 반환한다.\
$y_i$ : groud truth, $\hat{y_i}$ : prediction, $K_n$ : N개의 index 순열 함수, $c_i$ : class label, $b_i$ : box정보

$$ L_{Match}(y_i,\hat{y}_{{\sigma(i)}}) = -1_{c_i \neq \emptyset}\hat{p}_{{\sigma(i)}}(c_i) + 1_{c_i \neq \emptyset}L_{box}(b_i,\hat{b_i}) $$

$$ \hat{\sigma} = \argmax \limits_{\sigma \in K_N} \sum \limits_{i}^NL_{Match}(y_i,\hat{y}_{{\sigma(i)}}) $$

이때 $\sigma$는 prediction과 ground truth를 매칭하는 index function이다.

```math
L_{Hungarian}(y,\hat{y}) = \sum\limits_{i=1}^N -log \ \hat{p}_{{\hat{\sigma}(i)}}(c_i) + 1_{c_i \neq \emptyset}L_{box}(b_i,\hat{b_i})
```

$\hat{\sigma}$은 예측 결과와 ground truth를 최적의 이분매칭을 해주는 index 함수이다. 이 값을 통하여 class의 확률값과 bbox의 regression 값을 최종 loss로 
사용하였다.

---

## Architecture 
DETR의 구조는 크게 다음 3가지로 구분된다. 
- CNN backbone
- encoder-decoder transformer
- feed forward network

#### CNN backbone
Resnet, efficient net등 다양한 구조의 모델을 자유롭게 사용할 수 있다.
본 논문에서는 input image의 $w$, $h$ -> $\frac{w}{32},\frac{h}{32}$로 줄이면서 C값은 2048로 늘렸다.

#### Transformer encoder & decoder

<img src="../End-to-End Object Detection with Transformers/image/Fig3.jpeg">

cnn backbone으로부터 나온 feature에 대하여 1x1 conv를 통하여 C보다 작은 값의 d값에 대하여 새로운 feature map인 $z_0$를 추출한다. 이때 $z_0$는 d X h X w이다. 이 값을 transformer encoder 구조의 input으로 사용하기 위하여 d X hw로 변환한 후 position embeding 값을 추가하여 연산을 수행한다.

위 이미지는 self-attention연산을 통해 나온 값을 시각화 한 그림이다. 개별적인 instance에 대하여 attention이 잘 이루어지는 것을 확인할 수 있다.

encoder에서 나온 output을 학습 가능한 positional encoding값을 통하여 N개의 object_query에 대하여 decoder를 통하여 output을 반환한다.

#### Feed Forward network
최종적으로 fc-layer와 relu activation function을 사용하여 class의 정보와 box정보를 각 decoder의 qurey를 통하여 나온 output에 대하여 계산한다.

[paper](https://arxiv.org/abs/2005.12872)