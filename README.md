WT 공정 BIN/ITEM Para 기반 AUF Clustering System 구축

서울대 DS 이재진 교수 연구실


Abstract 

본 과제에서는 딥러닝을 이용한 wafer fail map 클러스터링 시스템을 구축하고자 한다. Wafer fail map은 불량의 원인에 따라 특성이 비슷하므로, 이를 군집화함으로써 품질 향상을 위한 유의인자를 발견할 수 있다. 이를 위해 최신 딥러닝 기반 클러스터링 방법들을 적용하고 결과를 비교해 본다. 첫 번째 접근법은 기존에 시도했던 latent representation 추출+클러스터링 방식에 data augmentation, random deletion등 새로운 전처리 기법을 적용하는 것이다. 두 번째 접근법은 SCAN, DeepCluster 등의 end-to-end 딥 클러스터링 모델을 활용하는 것이다. 



Introduction

한 장의 wafer에 발생한 불량을 이미지 형태로 나타낸 wafer fail map은 불량의 원인과 특성을 파악하기 위한 정보를 담고 있는 데이터이다. 반도체 생산공정의 특성상 fail은 서로 원인이 비슷한 것끼리 묶일 수 있으며, 이러한 군집화(clustering)를 통해 수율 및 품질 향상에 중요한 유의인자를 도출할 수 있다. 이에 따라 본 과제에서는 최근 우수한 성능을 보이는 딥러닝 기반 클러스터링 방법을 활용해 wafer fail map 클러스터링 시스템을 구축하고자 하였다.
본 과제에서 시도한 기존의 접근법은 데이터에서 latent representation을 추출한 뒤  클러스터링 알고리즘을 적용하는 두 단계로 나뉜다. Latent representation 추출을 위해 Autoencoder, Transformer 모델을 구현하고, 클러스터링 알고리즘으로는 Agglomerative clustering, Kmeans, HDBSCAN을 적용해 보았다. 
이후 새롭게 시도한 첫 번째 접근법은 단계를 나누지 않고 end-to-end 딥 클러스터링 모델을 활용하는 것이다. Deep clustering의 장점은 feature vector 추출, 차원 축소, 클러스터링의 각 단계를 통합함으로써 딥러닝 모델의 학습에 데이터의 특성과 클러스터링 성능을 동시에 반영할 수 있다는 점이다. 본 과제에서 활용한 Deep clustering 모델로는 SCAN 및 DeepCluster가 있다. 두 번째 접근법으로, data augmentation, random deletion 등의 data preprocessing을 통해 데이터 분포에 변화를 주어 클러스터링 성능을 향상시키고자 하였다.

Data preprocessing

대부분의 channel(?)에서 fail map의 분포가 불균형을 이루고 있다. 즉, fail이 적게 나타나는 wafer의 개수는 많고, fail이 많이 나타나는 wafer의 개수는 매우 적어서 클러스터링 성능에 영향을 미칠 수 있다. 이에 다음 두가지 방식의 전처리를 이용하여 데이터 분포의 불균형을 해소하고자 하였다. 
<fail이 적게 나타나는 wafer 예시> <fail이 많이 나타나는 wafer 예시>
아래 히스토그램의 x축은 한 wafer의 fail chip 개수이고, y축은 wafer의 개수이다. 앞서 언급한 데이터 분포의 불균형을 나타내고 있다. 
<데이터 불균형 histogram + data augmentation 그림, 예) channel 10>
Data augmentation
빨간 선의 아래에 있는 구간이 augmentation을 해주는 구간이며, 빨간 선만큼 증가시켜주었다. 이 방식을 이용하게 되면 빨간 선의 수준을 정하고, 빨간선을 기준으로 실제 몇배 늘려줄지 결정하는 hyperparameter들이 추가된다. 
Random deletion
Data augmentation 방식과는 반대로 빨간색 선의 위에 있는 구간의 data를 random하게 줄이는 방식이다. 이 방식을 이용하면 얼만큼 데이터를 줄일지 조절하기 위한 hyperparameter가 추가된다. 


Models

SCAN (추가 예정)
-Transformer + SCAN
-Autoencoder+SCAN


DeepCluster

개요
DeepCluster는 Facebook AI research에서 발표한 딥 클러스터링 알고리즘이다. 이는 label 없이도 데이터의 특성을 잘 담은 feature를 학습하기 위한 end-to-end unsupervised method로 요약할 수 있다. 주요 특징은 CNN을 통해 추출된 feature vector에 Kmeans 등의 clustering 알고리즘을 적용하고, 그 결과를 pseudo-label로 삼아 supervised learning과 유사하게 학습한다는 것이다. 
<Figure: DeepCluster 모델 구조>

학습의 한 사이클은 다음의 단계로 진행된다.

CNN을 통해 데이터의 특성을 추출하고 차원을 줄인 feature vector 학습
전체 feature vector에 PCA, normalization 등의 추가적인 처리 후 K-means 등의 clustering 알고리즘 적용
생성된 cluster assignment를 pseudo-label로 사용
CNN에 classification head를 추가한 후 pseudo-label과의 classification loss 계산
Backpropagation을 통해 CNN weights 업데이트

이후 DeepCluster 모델을 개선한 DeepCluster-v2 버전이 공개되었다. 이 버전에서 달라진 점들은 다음과 같다.

DeepCluster 모델은 매번 classification head를 새로 학습하는 등의 특성으로 인해 학습 과정이 불안정할 수 있다. v2 모델은 이를 개선하기 위해 classification head를 K-means 클러스터링 과정에서 얻어진 centroid 값들로 대체한다.
Multi-crop data augmentation 등 더 많은 data augmentation 방법을 사용한다.

2) Clustering 알고리즘
DeepCluster의 clustering 알고리즘으로 K-means, HDBSCAN, PIC(power iterative clustering)을 적용해 보았다. 
K-means는 데이터를 k개의 클러스터로 분류하기 위해 클러스터 형성과 centroid 업데이트를 반복하는 알고리즘이다. 
HDBSCAN은 밀도 기반 알고리즘으로, 각 데이터가 속한 영역의 밀도를 반영해  클러스터링하는 알고리즘이다.
PIC는 데이터의 similarity matrix에 power iteration을 적용해 저차원의 embedding을 찾아 클러스터링하는 알고리즘으로, spectral clustering의 한 종류로 볼 수 있다.

Results

Data preprocessing

두가지 방식의 효과를 분석하기 위해 기존 모델로 실험하였다. Auto-Encoder에서 추출된 latent representation에 hierarchical clustering을 적용한 방식이 성능 및 속도 측면에서 분석에 용이하기에 선택하였다. Baseline, data augmentation, random deletion 세가지 결과를 첨부한다. Data augmentation과 random deletion을 동시에 실험한 내용을 제외하였는데, 두가지 방식의 효과가 유사하기 때문이다. 실험 세팅은 다음과 같다. 
Clustering 개수 18 
클러스터당 데이터가 3개 미만이면 noise 처리 (처음 설정한 클러스터 개수보다 적을 수 있음)
Channel 10

Baseline vs Augmentation
<사진 첨부>
Baseline의 빨간 박스내의 wafer들이 augmentation시 사라지고, Augmentation의 빨간 박스 내의 wafer가 새로 생겼다. 이는 데이터의 분포를 변화시켜서 fail chip이 많이 나오는 경우를 자세히 분류하고자 하고, fail chip이 적게 나오는 경우를 덜 분류하는 것으로 보인다.
Baseline vs Random Deletion
<사진 첨부>
위와 마찬가지로 fail chip이 적게 나타나는 경우에 분류를 잘하는 경우와, fail chip이 많이 나타나는 경우에 분류를 잘하는 경우로 나뉜다. 
결과 분석
augmentation과 random deletion은 유사한 효과를 낼 수 있어, 모델에 잘 맞는 방식을 택하는 것이 좋다. 만약 성능이 비슷하다면 random deletion이 clustering하는 데이터의 개수가 적고, 조절해야하는 hyperparameter수가 적어 이를 선택하는 것이 좋다. 
그리고 위 결과는 데이터 분포에 따라 모델 성능의 trade-off를 보여준다. 즉, fail chip이 적게 나타나는 경우와 fail chip이 많이 나타나는 경우를 동시에 잘 분류하기는 어렵다. 그러므로 두가지 방식을 모두 이용한 결과를 보완적으로 이용하는 방식이 좋을 것 같다. 


  2. SCAN (추가 예정)

  3. DeepCluster 

DeepCluster 모델을 이용한 실험은 크게 두 가지로 나누어졌다. 첫째, K-means, HDBSCAN, PIC 등 다양한 클러스터링 알고리즘을 적용해 보았다. 또한 각각의 클러스터링 알고리즘의 특성에 맞게 모델의 구조를 수정하거나 hyperparameter를 조절해 보았다. 둘째, 모델의 성능 향상을 위해 위 Data preprocessing 항목의 data augmentation을 적용해 보았다.
Clustering 알고리즘
K-means
클러스터의 수 K=30으로 설정해 실험해 본 결과 (추가 예정)
DeepCluster-v2 모델의 변경사항을 반영해 classification head를 K-means centroid로 대체해 학습한 결과 (추가 예정)

HDBSCAN
DeepCluster 모델에 HDBSCAN을 적용해 본 결과, 클러스터의 수가 지나치게 많고 서로 특성이 비슷한 클러스터가 많아 사용하기 어렵다고 판단되었다. 이는 ~ (추가 예정) 때문인 것으로 보인다.
PIC (power iterative clustering)
세 가지 클러스터링 알고리즘 중, PIC을 적용한 모델의 성능이 metric 수치 및 직접 관찰했을 때 가장 좋은 결과를 보였다. 그러나 클러스터의 수가 약 800개로 현업에서 사용하기에 지나치게 많았다. 클러스터의 수를 줄이기 위한 변경사항 및 결과 (추가 예정)

Data augmentation (결과 추가 예정)


Conclusion
다양한 모델 및 전처리 방식을 실험해 본 결과, ~ (결과 정리후 추가 예정)

또한 본 과제에서 실험한 각각의 방법들은 각자의 장단점을 가지고 있으며, 채널(wafer test의 종류)에 따라 다른 결과를 보였다. 이는 반도체 생산공정 및 wafer fail map의 특성상 데이터의 특성과 분포가 채널에 따라 달라지기 때문이다. 따라서, 실제 wafer fail map 클러스터링 시스템을 구축하는 과정에서는 여러 가지 모델 및 전처리 방법을 비교하고, 채널의 특성에 맞는 방법을 선택하거나 여러 모델의 앙상블을 통해 성능을 향상시키는 방법으로 본 과제에서 얻은 결과를 활용할 수 있을 것이다.
