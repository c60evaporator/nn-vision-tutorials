# %% DummySin + LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import numpy as np

SEED = 42
TEST_SIZE = 0.1
BATCH_SIZE = 32
NUM_EPOCHS = 80  # number of epochs
DEVICE = 'cuda'  # 'cpu' or 'cuda'
PARAMS_SAVE_ROOT = '/scripts/params/classification'  # Directory for Saved parameters
SEQ_LENGTH = 40

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)
np.random.seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Create tensors for dataset
x = np.linspace(0, 499, 500)
eps = np.random.randn(500) * 0.2
y = np.sin(x * 2 * np.pi / 50) + eps
plt.plot(x, y)
plt.title('Dataset')
plt.show()
# Create Sequence data
def make_sequence_data(y, num_sequence):
    num_data = len(y)
    seq_data = []
    target_data = []
    for i in range(num_data - num_sequence):
        seq_data.append(y[i:i+num_sequence])
        target_data.append(y[i+num_sequence:i+num_sequence+1])
    seq_arr = np.array(seq_data)
    target_arr = np.array(target_data)
    return seq_arr, target_arr
y_seq, y_target = make_sequence_data(y, SEQ_LENGTH)
print(y_seq.shape)
print(y_target.shape)
# Separate Train and test data
num_val = int(TEST_SIZE * y_seq.shape[0])
y_seq_train, y_target_train = y_seq[:-num_val], y_target[:-num_val]
y_seq_val, y_target_val = y_seq[-num_val:], y_target[-num_val:]
a = x[SEQ_LENGTH:-num_val]
b = y_target_train[:,0]
plt.plot(x[SEQ_LENGTH:-num_val], y_target_train[:,0], label='Train')
plt.plot(x[-num_val:], y_target_val[:,0], label='Test')
plt.title('Train and Test target data')
plt.legend()
plt.show()
# Convert to Tensor (Should reshape from (sample, seq) to (seq, sample, input_size))
y_seq_t = torch.FloatTensor(y_seq_train).permute(1, 0).unsqueeze(dim=-1)
y_target_t = torch.FloatTensor(y_target_train).permute(1, 0).unsqueeze(dim=-1)
y_seq_v = torch.FloatTensor(y_seq_val).permute(1, 0).unsqueeze(dim=-1)
y_target_v = torch.FloatTensor(y_target_val).permute(1, 0).unsqueeze(dim=-1)

###### 2. Define Model ######
# Define Model (https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/)
class LSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x_last = x[-1]
        output = self.linear(x_last)
        return output

model = LSTM(100)
print(model)
# Send model to GPU
model.to(device)

###### 3. Define Criterion & Optimizer ######
criterion = nn.MSELoss()  # Criterion (MSE)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer (Adam)

###### 4. Training ######
losses = []  # Array for storing loss (criterion)
val_losses = []  # Array for validation loss
start = time.time()  # For elapsed time
# Epoch loop
for epoch in range(NUM_EPOCHS):
    # Send data to GPU
    y_seq_t_minibatch = y_seq_t.to(device)  
    y_target_t_minibatch = y_target_t.to(device)
    # Update parameters (No minibatch)
    optimizer.zero_grad()  # Initialize gradient
    output = model(y_seq_t_minibatch)  # Forward (Prediction)
    loss = criterion(output, y_target_t_minibatch)  # Calculate criterion
    loss.backward()  # Backpropagation (Calculate gradient)
    optimizer.step()  # Update parameters (Based on optimizer algorithm)
    # Calculate running losses
    losses.append(loss.item())

    # Calculate validation metrics
    y_seq_v_minibatch = y_seq_v.to(device)
    y_target_v_minibatch = y_target_v.to(device)
    val_output = model(y_seq_v_minibatch)  # Forward (Prediction)
    val_loss = criterion(val_output, y_target_v_minibatch)  # Calculate criterion
    val_losses.append(val_loss.item())

    print(f'epoch: {epoch}, loss: {loss}, val_loss: {val_loss}, elapsed_time: {time.time() - start}')

###### 5. Model evaluation and visualization ######
# Plot loss history
plt.plot(losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Loss history')
plt.legend()
plt.show()
# Prediction of test data
y_pred = model(y_seq_v.to(device))
plt.plot(x, y, label='true')
plt.plot(np.arange(500-num_val, 500), y_pred.detach().to('cpu'), label='pred')
plt.xlim([400, 500])
plt.legend()
plt.show()

###### 6. Save the model ######
# Save parameters
params = model.state_dict()
if not os.path.exists(PARAMS_SAVE_ROOT):
    os.makedirs(PARAMS_SAVE_ROOT)
torch.save(params, f'{PARAMS_SAVE_ROOT}/dummysin_lstm.prm')
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/dummysin_lstm.prm')
model_load = LSTM(100)
model_load.load_state_dict(params)

# %%
