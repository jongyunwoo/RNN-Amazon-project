#필요한 라이브러리 호출
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm_notebook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("/Users/ujong-yun/Documents/GitHub/project/Amazon_dataframe.csv")
#날짜 칼럼을 인덱스로 사용
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

#데이터 형식 변경
data['Volume'] = data['Volume'].astype(float)
data['OBV'] = data['OBV'].astype(float)
#훈련과 레이블 분리
columns_to_exclude = ['Close']  # 제외할 열 이름
X = data.iloc[:, ~data.columns.isin(columns_to_exclude)]  # ~ 연산자로 해당 열을 제외
Y = data.iloc[:, 4:5]

#데이터 분포 조정
ss = StandardScaler()

X_ss = ss.fit_transform(X)
Y_ss = ss.fit_transform(Y)

X_train = X_ss[:201, :]
X_test = X_ss[201:, :]

Y_train = Y_ss[:201, :]
Y_test = Y_ss[201:, :]

print("Training Shape", X_train.shape, Y_train.shape)
print("Testing Shape", X_test.shape, Y_test.shape)

#데이터셋의 형태 및 크기 조정
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

Y_train_tensors = Variable(torch.Tensor(Y_train))
Y_test_tensors = Variable(torch.Tensor(Y_test))

X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print("Training Shape", X_train_tensors_f.shape, Y_train_tensors.shape)
print("Testing Shape", X_test_tensors_f.shape, Y_test_tensors.shape)

#LSTM 네트워크
class LSTM(nn.Module):
  def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, dropout):
    super(LSTM, self).__init__()
    self.num_classes = num_classes
    self.num_layers = num_layers
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.seq_length = seq_length

    self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)
    self.dropout = nn.Dropout(dropout)
    self.fc_1 =  nn.Linear(hidden_size, 128)
    self.fc = nn.Linear(128, num_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
    h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
    c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
    output, (hn, cn) = self.lstm(x, (h_0, c_0))
    hn = hn.view(-1, self.hidden_size)
    out = self.dropout(hn) 
    out = self.relu(out)
    out = self.fc_1(out)
    out = self.relu(out)
    out = self.fc(out)
    return out
  

#결정계수 계산 정의
def r_squared(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


from sklearn.model_selection import ParameterGrid
  
param_grid = {
    'dropout': [0.1, 0.2, 0.3],
    'num_epochs': [1000, 1500, 2000],
    'hidden_size': [5, 10, 15],
    'learning_rate': [0.001, 0.01, 0.0001]
}
input_size = 57
num_layers = 1
num_classes = 1
best_params = None
best_r2 = float('-inf')

# 그리드 서치 실행
for params in ParameterGrid(param_grid):
    dropout = params['dropout']
    num_epochs = params['num_epochs']
    hidden_size = params['hidden_size']
    learning_rate = params['learning_rate']
    
    model = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1], dropout)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model.forward(X_train_tensors_f)
        optimizer.zero_grad()
        loss = criterion(outputs, Y_train_tensors)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    # 예측값 계산
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensors_f)
        test_outputs = model(X_test_tensors_f)

    #스케일링 되돌리기
    train_outputs = ss.inverse_transform(train_outputs)
    test_outputs = ss.inverse_transform(test_outputs)
    Y_train = ss.inverse_transform(Y_train_tensors)
    Y_test = ss.inverse_transform(Y_test_tensors)

    r2 = r_squared(Y_test, test_outputs)
    
    if r2 > best_r2:
        best_r2 = r2
        best_params = params

print("Best R-squared:", best_r2)
print("Best Parameters:", best_params)

# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(Y_train)), Y_train, label='Train True')
plt.plot(np.arange(len(train_outputs)), train_outputs, label='Train Predicted')
plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), Y_test, label='Test True')
plt.plot(np.arange(len(train_outputs), len(train_outputs) + len(test_outputs)), test_outputs, label='Test Predicted')
plt.title('Predicted vs True')
plt.xlabel('Data Points')
plt.ylabel('Value')
plt.legend()
plt.show()

