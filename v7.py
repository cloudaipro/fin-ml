import pandas as pd
import numpy as np
import pandas_datareader.data as web
from matplotlib import pyplot
import seaborn as sns

# hyperparameters turning
from ray import tune, train, ray
from ray.tune.schedulers import ASHAScheduler
ray.init(log_to_driver=False)

#Plotting 
from pandas.plotting import scatter_matrix

#Libraries for Statistical Models
import statsmodels.api as sm

#Diable the warnings
import warnings
warnings.filterwarnings('ignore')

import os

pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = False

_TARGET_STK = 'MSFT'
data_dir = './data/'

from datetime import datetime
import yfinance as yfin

# Loading the data
stk_tickers = [_TARGET_STK]
idx_tickers = ['VIXCLS', 'SP500', 'DJIA']

start = datetime(2014, 1, 1)
end = datetime(2023, 12, 31)

stk_file = f"{data_dir}{_TARGET_STK}.csv"
if os.path.isfile(stk_file):
    stk_data = pd.read_csv(stk_file).set_index('Date')
    print(f"read {stk_file} completely!")
else:
    # stk_data = web.get_data_yahoo(stk_tickers, start, end)
    stk_data = yfin.download(stk_tickers, start, end).dropna()
    stk_data.to_csv(stk_file)
    print(f"download {_TARGET_STK} from yfin and write to {stk_file} completely!")
    
idx_file = f"{data_dir}{'_'.join(idx_tickers)}.csv"
if os.path.isfile(idx_file):
    idx_data = pd.read_csv(idx_file).set_index('DATE')
    print(f"read {idx_file} completely!")
else:
    idx_data = web.DataReader(idx_tickers, 'fred', start, end).dropna()
    idx_data.to_csv(idx_file)
    print(f"download {idx_tickers} from yfin and write to {idx_file} completely!")

print(stk_data)
print(idx_data)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
validation_size = 0.2
epoch_num = 300
batch_size = 32
log_dir_base = f'{os.getcwd()}/runs/{_TARGET_STK}'
log_dir = log_dir_base
print(log_dir)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class LSTMDataSet(Dataset):
    def __init__(self, X, Y, seq_len):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
    def __len__(self):
        return len(self.X) - self.seq_len + 1
    
    def __getitem__(self, idx):
        return (torch.tensor(np.array(self.X[idx: idx + self.seq_len]), dtype=torch.float32),
                torch.tensor(np.array(self.Y.iloc[idx + self.seq_len - 1,:]), dtype=torch.float32))

from torch import nn
class StockPricePredictionLSTM(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, num_layers, num_fc_layers, activation_type):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        """
            input_size    : The number of expected features in the input x
            hidden_size   : The number of features in the hidden state h
            num_layers    : Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
            bias          : If False, then the layer does not use bias weights b_ih and b_hh. Default: True
            batch_first   : If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
            dropout       : If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
            bidirectional : If True, becomes a bidirectional LSTM. Default: False
            proj_size     : If > 0, will use LSTM with projections of corresponding size. Default: 0
        """
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        layers = []
        in_features = self.hidden_size
        for i in range(1, num_fc_layers):
            out_features = int(in_features / 2)
            if (out_features <= 1):
                break
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU() if activation_type == 1 else
                          nn.Sigmoid()) if activation_type == 2 else nn.Tanh()
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*layers)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            initrange = 0.5
            nn.init.uniform_(m.weight, -initrange, initrange)
            nn.init.zeros_(m.bias)
            # print(f"{m.in_features},{m.out_features}")

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        out, (h_out, _) = self.rnn(x, (h_0, c_0))

        fc_input = h_out[-1].view(-1, self.hidden_size)
        return self.fc(fc_input)

import math 
from sklearn.metrics import mean_squared_error

def eval_dl_method(model, dl, criterion=None, device=device):
    model.eval()
    y_gt = []
    y_pred = []
    vloss = 0.0
    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        if criterion != None:
            vloss += criterion(outputs, y).item()
        # print(f"{i}:{vloss}")
        y_gt.extend(y.cpu().detach().numpy().reshape(-1))
        y_pred.extend(outputs.cpu().detach().numpy().reshape(-1))
    
    return (math.sqrt(mean_squared_error(y_gt, y_pred)) if np.isnan(y_pred).any() == False else 9999, y_gt, y_pred, vloss)
    
def prepare_buy_sell_signal():
    buy_sell_signal = stk_data[[]].copy()
    sma = pd.concat([stk_data.ta.sma(close='Adj Close', length=10), stk_data.ta.sma(close='Adj Close', length=60)], axis=1).dropna()
    buy_sell_signal['Signal'] = (sma['SMA_10'] > sma['SMA_60']).astype('float32')
    return buy_sell_signal

def gen_analysis_data(stock_name, return_period):
    import pandas_ta
    data = stk_data
    
    data = pd.concat([data.ta.percent_return(length=return_period, prefix=stock_name),
                        data.ta.adosc(prefix=stock_name),
                        data.ta.kvo(prefix=stock_name), 
                        data.ta.rsi(close='Adj Close', length=10, prefix=stock_name)/100,
                        data.ta.rsi(close='Adj Close', length=30, prefix=stock_name)/100,
                        data.ta.rsi(close='Adj Close', length=200, prefix=stock_name)/100,
                        data.ta.stoch(k=10, prefix=stock_name)/100,
                        data.ta.stoch(k=30, prefix=stock_name)/100,
                        data.ta.stoch(k=200, prefix=stock_name)/100],
                        axis=1)
    data = data.dropna().astype('float32')
    return data, data.columns[:-9].values, data.columns[-9:].values

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn_pandas import DataFrameMapper

def prepare_dataloader(return_period, seq_len):
    Target_Data, un_standardized, standardized = gen_analysis_data(_TARGET_STK, 5)
    buy_sell_signal = prepare_buy_sell_signal()

    standardized = np.append(standardized, buy_sell_signal.columns.values)

    X1 = pd.concat([Target_Data, buy_sell_signal], axis=1)
    X3 = pd.concat([idx_data['SP500'].pct_change(return_period), idx_data['DJIA'].pct_change(return_period)], axis=1)
    X3 = X3.rename(columns={ column : f"{column}_PCTRET_{return_period}" for column in X3.columns.to_list()})
    X3 = pd.concat([idx_data['VIXCLS'], X3], axis=1).dropna()
    X1_X3 = pd.concat([X1, X3], axis=1).dropna()

    Y_raw = stk_data.loc[:, 'Adj Close'].to_frame().pct_change(return_period).shift(-return_period).dropna().astype('float32')
    Y_raw.columns = [f"{_TARGET_STK}_pred_{return_period}"]

    dataset = pd.concat([X1_X3, Y_raw], axis=1).dropna()

    X = dataset.loc[:, X1_X3.columns]
    Y = dataset.loc[:, Y_raw.columns]

    train_size = int(len(X) * (1 - validation_size))
    X_train = X.iloc[0:train_size]
    Y_train = Y.iloc[0:train_size]
    X_test  = X.iloc[train_size - seq_len + 1:len(X)]
    Y_test  = Y.iloc[train_size - seq_len + 1:len(Y)]

    features = [([column], StandardScaler()) for column in un_standardized]
    features.extend([([column], StandardScaler()) for column in X3.columns.values])
    features.extend([([column], None) for column in standardized])
    X_dfm = DataFrameMapper(        
        features,
        input_df=True, df_out=True)
    X_train = X_dfm.fit_transform(X_train)
    X_test = X_dfm.transform(X_test)

    train_loader = DataLoader(LSTMDataSet(X_train, Y_train, seq_len), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(LSTMDataSet(X_test, Y_test, seq_len), batch_size=batch_size)

    return train_loader, test_loader
    
def save_model(model, config, file_path):
    state = {
        'time': str(datetime.now()),
        'model_state': model.state_dict(),
        'input_size': model.input_size,
        'config': config            
    }
    torch.save(state, file_path)

def load_model(file_path):
    data_dict = torch.load(file_path)
    config = data_dict['config']    
    model = StockPricePredictionLSTM(config['seq_len'], input_size=data_dict['input_size'],
                                     hidden_size=int(config['hidden_size']),num_layers=int(config['num_layers']),
                                     num_fc_layers=int(config['num_fc_layers']), activation_type=int(config['activation_type']))

    model.load_state_dict(data_dict['model_state'])
    return model, config

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

def do_train(model, optimizer, train_dl, test_dl, id_str, config, writer):
    criterion = torch.nn.MSELoss()

    model_name = f"{log_dir}/{id_str}.pt"
    best_loss = 999999999    
    total_loss = 0.0
    total_vloss = 0.0
    for epoch in tqdm(range(epoch_num)):
        model.train()
        running_loss = 0.0
        
        for i, (x, y) in enumerate(train_dl):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_model(model, config, model_name)

        with torch.no_grad():
            (testScore, test_y_gt, test_y_pred, running_vloss)  = eval_dl_method(model, test_dl, criterion)
            train.report({"mse_score":testScore})
            
        writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : running_loss / len(train_dl), 'Validation' : running_vloss / len(test_dl) },
                            epoch + 1)
        total_loss += running_loss / len(train_dl)
        total_vloss += running_vloss / len(test_dl)

        writer.flush()
        
    return {'Train loss':total_loss/epoch_num, 'Validation loss': total_vloss/epoch_num}

def train_LSTM(config):
    return_period = config["return_period"]
    seq_len = config["seq_len"]
    lr = config["lr"]
    momentum = config["momentum"]
    optim_type = config["optim_type"]
    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]
    num_fc_layers = config["num_fc_layers"]
    activation_type = config["activation_type"]

    id_str = "_".join(str(v) if v < 1 else f'{v:g}' for v in config.values())
    print(id_str)
    writer = SummaryWriter(f"{log_dir}/{id_str}")    

    train_loader, test_loader = prepare_dataloader(return_period, seq_len)
    model = StockPricePredictionLSTM(seq_len, input_size=len(train_loader.dataset.X.columns),
                                     hidden_size=hidden_size,num_layers=num_layers,
                                     num_fc_layers=num_fc_layers, activation_type=activation_type)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optim_type == 1 else torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    metric_dict = do_train(model, optimizer, train_loader, test_loader, id_str, config, writer)
    writer.add_hparams(
        config,
        metric_dict
    )
    writer.close()

search_space = {
    "return_period": tune.grid_search([3, 5]), #[2,3,5,10]
    "seq_len": tune.grid_search([3, 5, 10]),
    "lr": tune.grid_search([0.01]), #[0.001, 0.01, 0.1]
    "momentum": tune.uniform(0.1, 0.9),
    "optim_type": tune.grid_search([2]), # [1, 2]
    "num_layers": tune.grid_search([2, 4, 8, 16]), #[1, 2, 4, 8]
    "hidden_size": tune.grid_search([32, 64, 128, 256]), #[8, 16, 32, 64, 128]
    "num_fc_layers": tune.grid_search([1, 2, 3]), #1, 2, 3]),
    "activation_type": tune.grid_search([1, 2, 3]) #, 2, 3])
}

turning_parameters = []
total_configs = 1
for k, v in search_space.items():
    if type(v).__name__ == 'dict' and list(v.keys())[0] == 'grid_search' and len(list(v.values())[0]) > 1:
        turning_parameters.append(k)
        total_configs *= len(list(v.values())[0])
print(turning_parameters)
print(f"Total count of configs = {total_configs}")


import warnings
warnings.filterwarnings('ignore', category=Warning)

time_str = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
log_dir = f'{log_dir_base}/{time_str}'

analysis = tune.run(train_LSTM, 
                    config=search_space,
                    resources_per_trial={'cpu':0.125, 'gpu':0.125},
                    metric="mse_score",
                    mode="min")

mse_list = []
trial_list = list(analysis.trial_dataframes.values())
for i, trial in enumerate(trial_list):
    # if trial.empty == False:
    d = pd.DataFrame.from_dict({"mse_score": trial.describe().loc['mean', 'mse_score'], "trial_id": trial.loc[0:0,'trial_id'] })
    # else:
    #     d = pd.DataFrame.from_dict({"mse_score": [np.NaN], "trial_id": [np.NaN]})
    mse_list.append(d)
mse_df = pd.concat(mse_list)
mse_df = mse_df.reset_index().loc[:, ["mse_score", "trial_id"]]
print(mse_df)

import shutil

config_df = pd.DataFrame(analysis.get_all_configs().values())
# print(config_df)
results = pd.concat([mse_df, config_df], axis=1)
# print(results)

sorted_results = results.sort_values(by="mse_score")
# print(sorted_results.head(100))
sorted_results_file = f"{_TARGET_STK}_sorted_results.csv"
sorted_results.to_csv(sorted_results_file)

best_config = config_df.iloc[sorted_results.index[0]].to_dict()
id_str = "_".join(str(v) if v < 1 else f'{v:g}' for v in best_config.values())
best_model_name = f"{log_dir}/{id_str}.pt"
print(best_model_name)
shutil.copy(best_model_name, f"{_TARGET_STK}.pt")

sorted_results_file = f"{_TARGET_STK}_sorted_results.csv"
sorted_results = pd.read_csv(sorted_results_file, dtype='str')
best_config = sorted_results.loc[0]
print(best_config)

import math 
from sklearn.metrics import mean_squared_error

pd.set_option('display.precision', 5)

model, config = load_model(f"{_TARGET_STK}.pt")
model.to(device)

train_loader, test_loader = prepare_dataloader(config["return_period"], config["seq_len"])
model.eval()

(trainScore, train_y_gt, train_y_pred, _)= eval_dl_method(model, train_loader, device=device)
(testScore, test_y_gt, test_y_pred, _)  = eval_dl_method(model, test_loader, device=device)
print(test_y_pred)
print(f"Train RMSE: {trainScore:.2f}\nTest RMSE: {testScore:.5f}")

target_raw = stk_data.loc[test_loader.dataset.X.index.values]
print(target_raw.head)
target_raw = target_raw.drop(target_raw.index[range(config["seq_len"] - 1)], axis=0)

test_y_pred_df = pd.DataFrame(index=target_raw.index.copy())
test_y_pred_df['pred_price'] = test_y_pred
test_y_pred_df['pred_price'] = (test_y_pred_df['pred_price'] + 1) * target_raw['Close']
test_y_pred_df = test_y_pred_df.shift(config["return_period"])
tmp_data = pd.concat([target_raw, test_y_pred_df], axis=1).dropna()
tmp_data['Close'].plot()
tmp_data['pred_price'].plot()
pyplot.legend()
pyplot.show()

target_raw = stk_data.loc[test_loader.dataset.X.index.values]
# print(target_raw.head(10))
target_raw = target_raw.drop(target_raw.index[range(config["seq_len"] - 1)], axis=0)
# print(target_raw.head(10))

test_y_pred_df = pd.DataFrame(index=target_raw.index.copy())
test_y_pred_df['pred_price_change'] = test_y_pred
# print(test_y_pred_df.head(10))
test_y_pred_df['pred_price'] = (test_y_pred_df['pred_price_change'] + 1) * target_raw['Close']
# print(test_y_pred_df.head(10))
# print(test_y_gt[:10])
test_y_pred_df = test_y_pred_df.shift(config["return_period"])
# print(test_y_pred_df.head(10))
tmp_data = pd.concat([target_raw, test_y_pred_df], axis=1).dropna()
tmp_data['Close'].iloc[:20].plot()
tmp_data['pred_price'].iloc[:20].plot()
pyplot.legend()
pyplot.show()

