# 아마존 주식 가격 예측
코로나19 발생 이후 한국 주식시장에서 개인 투자자의 참여가 크게 늘었다. 하지만 전문성이 높은 기관 투자자와 외국인 투자자 사이에서 개인 투자자들은 높은 수익률을 가져오기는 어렵다.
나는 개인 투자자들의 수익률을 조금이라도 올리기 위해 주식 데이터를 시계열 데이터로 사용하여 deep learning을 통해 미래의 가격을 예측하고 이를 통해 수익을 좀 더 쉽게 실현할 수 있도록 하겠다.
LSTM모델과 GRU모델을 설계하여 같은 범위의 hyper-parameter 중에서 Grid- Search을 통해 각 모델의 최적의 hyper-parameter구한다. 이후 결정계수를 평가지표로 하여 LSTM모델과 GRU모델 중 어느 모델이 주가를 예측하는데 적합한지 판단한다.
1. Dataset
야후파이낸스에서 제공하는 데이터를 사용한다. yfinance라이브러리를 통해 아마존의 1년치 주가 데이터를 가져온다. 실제로는 3, 5, 20, 60, 120,일 이동평균선을 구하기 위해 2022-05-08부터 2023-12-08데이터를 가져온다. 이동평균선을 구한 후 2022-12-08부터 2023-12-08데이터를 training 및 test에 사용한다. 아마존 수익률을 입력 값으로 넣기 위해 (2)식을 이용해 수익률을 구한다. 나스닥 변동률을

수익률 = ((다음 날 종가 - 오늘 종가) / 오늘 종가) * 100 (2)

입력 값으로 사용하기 위해 2022-12-08부터 2023-12-08 나스닥 데이터를 yfinance을 통해 불러온다. (2)식을 이용하여 나스닥 일일 변동률을 구한다. 아마존의 수익률과 나스닥의 변동률이 stational한 데이터인지 확인하기 위해 Dickey-Fuller검증을 진행한다. 아마존 주가의 3, 5, 20, 60, 120 이동평균선을 구한다.
이후 주식 시장에서 사용하는 다양한 기술적 지표를 보조지표로 사용한다. DPO, CCI, MFI, ADI, OBV, CMF, FI, EOM(EMV), VPT, NVI, VMAP, ATR, BHB, BLB, KCH,  KCM, DCH, DCL, DCM, UI, SMA, EMA, WMA, MACD, ADX, -VI, +VI, TRIX, MI,  KST, Ichimoku, Parabolic SAR, STC, RSI, SRSI, TSI, UO, SR, WR, AO, KAMA, ROC, PPO, PVO을 보조지표로 사용한다. 이에 대한 설명은 Reference 파트 뒤[참고]에 첨부하겠다.
위에 언급한 데이터들과 Open, High, Low, Adjust Close, Volume을 포함하여 총 57개의 feature을 입력 값으로 사용하고 Close를 예측 값으로 사용한다.

2. Model
-LSTM, [3] "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.”
(그림5)LSTM은 장단기 메모리(Long Short-Term Memory)의 약자로, 순환 신경망(Recurrent Neural Network, RNN)의 한 종류이다.
그러나 기본 RNN은 시퀀스가 길어질수록 장기 의존성 문제(long-term dependencies problem)를 겪는 경향이 있다. 이런 문제 때문에 RNN은 긴 시퀀스를 처리하기에 적합하지 않을 수 있다. 이를 극복하기 위해 나온 것이 LSTM이다.
LSTM은 RNN의 변형으로서, 시퀀스 데이터의 장기 의존성 문제를 해결하기 위해 고안되었습니다. 주요 구성 요소는 다음과 같다.
1. Cell State (기억 셀): 정보를 저장하고 전달하는 핵심 부분으로, 장기적인 의존성을 학습하는 데 중요하다. 정보의 삭제와 추가가 이루어진다.
2. 입력 게이트(Input Gate): 셀 상태에 새로운 정보를 추가하기 위한 게이트. 어떤 정보를 업데이트할 지 결정한다.
3. 망각 게이트(Forget Gate): 불필요한 정보를 삭제하는 게이트. 이전 기억 중에서 어떤 정보를 버릴지 결정한다.
4. 출력 게이트(Output Gate): 셀의 상태를 바탕으로 새로운 숨겨진 상태(hidden state)를 만들고 이를 출력한다.
 (그림5)
<img width="419" alt="image" src="https://github.com/jongyunwoo/project/assets/127372349/b94487b2-7a6e-48ad-9d61-a3e674c0fe44">





-GRU, [4]Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
(그림6)GRU는 LSTM의 변종 중 하나로, 장기 의존성 문제를 해결하기 위해 고안된 순환 신경망의 한 종류.
GRU는 이를 해결하기 위해 LSTM보다 더 간결한 구조를 갖고 있다. LSTM과 비교했을 때 GRU는 기억 셀과 잊기 게이트를 하나의 셀로 합친 형태를 가지고 있다. 이러한 단순화된 구조로 인해 GRU는 LSTM보다 더 적은 파라미터를 가지고 있고, 따라서 학습 시간이 조금 더 짧을 수 있다.
GRU의 주요 구성 요소는 업데이트 게이트(Update Gate)와 재설정 게이트(Reset Gate).
1. 업데이트 게이트: 기존 정보를 얼마나 유지할지를 결정한다. 이 게이트는 얼마나 많은 이전 정보를 보존할지를 결정하며, 이를 통해 기존 정보 중에서 어떤 정보를 유지하고 어떤 정보를 버릴지를 결정한다.
2. 재설정 게이트: 새로운 입력 값과 얼마나 조합할지를 결정한다. 이 게이트는 새로운 입력과 기존의 은닉 상태의 얼마나 많은 정보를 섞어서 사용할지를 결정한다.
 (그림6)
<img width="354" alt="image" src="https://github.com/jongyunwoo/project/assets/127372349/d23cd2bc-e65d-47fa-9bfe-3e5627f2d0dc">

4. Hyper-parameter튜닝
LSTM모델과 GRU모델 모두 같은 범위[표1]의 hyper-parameter중에서 Grid-Search을 통해 각각 최적의 hyper-parameter을 찾는다. 과접합을 피하기 위해 dropout을 적용한다.
[표1]
LSTM, GRU			
Num_epochs	1000	1500	2000
Hidden_layers	5	10	15
Learning_rate	0.01	0.001	0.0001
Dropout	0.1	0.2	0.3
Num_layers	1	1	1
Num_classes	1	1	1



5. Evaluation metrics
평가지표는 결정 계수(R-squared)을 사용한다. 결정 계수 (R-squared)는 주어진 데이터에 대한 회귀 모델의 적합도를 측정하는 지표이다. 주어진 종속 변수에 대한 모델의 설명력을 나타낸다. 일반적으로 R-squared는 식(3)과 같이 계산된다.
 (3)<img width="141" alt="image" src="https://github.com/jongyunwoo/project/assets/127372349/f237f277-1287-42fc-9d50-dd95e16395f2">

여기서 SSres는 잔차 제곱의 합, SStot는 총 변동의 합이다. R제곱에서 루트를 씌우면 R을 구할 수 있다. R값이 1에 가까울 수록 모델에 적합하다고 할 수 있다.

6. Result
LSTM의 R-squared는 dropout = 0.1, hidden_size = 15, learning_rate = 0.0001, num_epochs = 2000에서 약 0.8540값을 나타낸다(그림7).
GRU의 R-squared는 dropout = 0.3, hidden_size =. 15, learning_rate =0.0001, num_epochs = 1000에서 약 0.9591값을 나타낸다(그림8).
<img width="415" alt="image" src="https://github.com/jongyunwoo/project/assets/127372349/bcccebaa-61ab-436a-9fed-913025e3d1f2">(그림7).
 <img width="415" alt="image" src="https://github.com/jongyunwoo/project/assets/127372349/4009e980-1d3e-41fe-be7a-acf34a20e68d">(그림8)
각 모델 별 hyper-parameter가 적용된 예측 그래프를 보면 GRU모델(아래)이 아마존 종가 추세를 LSTM모델(위)보다 상대적으로 잘 맞추는 것을 확인할 수 있다(그림9), (그림10). 또한 R-squared값을 확인하면 GRU모델이 더 적합하다는 것을 확인할 수 있다.
<img width="367" alt="image" src="https://github.com/jongyunwoo/project/assets/127372349/a37485d0-3f2e-4c0c-a350-5ff251071738"> (LSTM모델, 그림9)
<img width="354" alt="image" src="https://github.com/jongyunwoo/project/assets/127372349/697a1c58-c4cd-442f-b3d9-41fa87b09816"> (GRU모델, 그림10)



