# Mean teachers are better role models

## Abstract
[Temporal Ensembling](https://arxiv.org/abs/1610.02242)은 semi-supervise learning 분야에서 2017년 기준 몇몇 Banchmark에서 sota의 결과를 보여주고 있었다. 하지만 큰 데이터셋에서 다루기 어려워 진다는 단점을 갖고 있다. \
이러한 문제를 극복한 방법이 Mean Teacher method이다.

---
## Π-MODEL
[Temporal Ensembling](https://arxiv.org/abs/1610.02242)논문에서 등장하는 모델인 Π-MODEL에 대하여 먼저 알아보자.

<img src="../Mean teachers are better role models/Images/Π-MODEL.jpeg">

$ x_i : input \newline y_i : label \newline (z_i, \tilde{z_i}) : output1,\ output2 $ 

Π-MODEL은 input $x_i$에 대하여 확률적인 augmentation과 dropout이 포함된 네트워크를 통과하여 서로다른 output $z_i$와 $\tilde{z_i}$를 반환한다. 이를 이용하여 2가지의 loss를 구하게 된다.
- supervised loss : Cross Entropy($y_i$, $z_i$)
- unsupervised loss : MSE($z_i$, $\tilde{z_i}$)


이때 Loss의 scale을 위하여 t번째 학습 epoch에 대하여 종속적으로 증가하는 $w(t)$를 사용한다. \
최종적인 Loss는 다음과 같이 결정하여 모델을 학습하게 된다.

- Loss = supervised loss + $w(t)$ X unsupervised loss

---
## Temporal ensembling
<img src="../Mean teachers are better role models/Images/Temporal Ensembling.jpeg">

$ x_i : input \newline y_i : label \newline (z_i,\tilde{z_i}) : output,\ ensemble \ output $

Temporal ensembling은 Π-MODEL과 다른점은 다음과 같다.
- input에 대하여 하나의 output을 추론한다.
- 이전의 network에서 추론한 결과를 사용한다.

momentum term $\alpha$와 training epoch t에 대하여 $\tilde{z_i}$를 다음과 같이 구한다.\
 $Z_i = \alpha Z_{i-1} + (1-\alpha)z_i$ \
 $\tilde{z_i} = (1-\alpha^t)Z_i$

Temporal ensembling의 Loss는 Π-MODEL의 Loss와 동일한 방법으로 계산하여 결정한다.
 - Loss = supervised loss + $w(t)$ X unsupervised loss

Π-MODEL과 비교하였을때 2가지 장점을 갖는다.
- input에 대하여 하나의 output을 생성하므로 훈련과정내 inference time이 2배정도 빠르다.
-  $\tilde{z_i}$에 대하여 더 적은 noise가 발생한다.

---
## Mean Teacher
<img src="../Mean teachers are better role models/Images/Mean Teacher.jpeg">

[Temporal Ensembling](https://arxiv.org/abs/1610.02242)은 input에 대하여 현재의 network와 이전의 network의 output을 이용하여 모델의 성능을 개선하였다. \
만약 데이터셋의 크기가 커진다면 이전의 network로부터 생성된 output이 update되는 속도는 느려진다. 이로인해 현재 network와 이전의 network로 생성된 output을 비교하는 것을 loss로 사용하는 것은 부적절한 방법이 될 수 있다.

Mean Teacher는 Temporal Ensembling의 한계를 극복하기 위해 이전의 network로 부터의 output을 사용하는 것 대신에 model weight의 평균을 사용한다.

Mean Teacher방법은 추가적인 학습없이 Student model로부터 Teacher model을 학습하려 한다.


- Train Flow
1. input에 대하여 student model, teacher model 추론
2. student model 학습 및 weight update
3. 학습된 student model로 부터 EMA를 활용하여 teacher model weight update

#### sutdent model

student model을 학습하기 위하여 Loss를 다음과 같이 설정한다.
$Loss = Classification\ Loss + Consistency\ Loss $

- $Classification\ Loss$ : output과 target에 대한 Cross entropy Loss를 사용했다.

- $Consistency\ Loss$ : student model과 teacher model의 2가지 output에 대하여 Mse Loss 혹은 KL-Divergence를 사용한다.
본 논문에서는 Mse Loss를 사용했다.

#### Teacher model

student model의 weight $\theta_t$에 대하여 teacher model의 weight $\theta_t'$을 다음과 같이 update한다. 

$\theta_t = \alpha \theta_t' + (1-\alpha)\theta_t$

---


<img src="../Mean teachers are better role models/Images/Table3.jpeg">

테이블 좌측으로부터 unlabeled data를 각 0,100000,500000개를 추가한 결과이다.

결과를 보면 MeanTeacher 방법을 사용하였을때 Π-MODEL과 비교하여 성능이 향상한 것을 확인할 수 있다.

[Paper Link](https://arxiv.org/pdf/1703.01780.pdf)