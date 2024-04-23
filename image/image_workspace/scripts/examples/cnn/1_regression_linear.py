# %% Linear Regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt

SEED = 42
TEST_SIZE = 0.25
NUM_EPOCHS = 500  # number of epochs
plt.style.use('ggplot')

###### 1. Create dataset & Preprocessing ######
# Create tensors for dataset
torch.manual_seed(SEED)
a =3 
b = 2
x = torch.linspace(0, 5, 100).view(100, 1)
eps = torch.randn(100,1)
y = a * x + b + eps
plt.scatter(x, y)
plt.title('Dataset')
plt.show()
# Create dataset
# class TensorDatasetRegression(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#     def __len__(self):
#         return len(self.x)
#     def __getitem__(self, idx):
#         sample = {
#             'feature': torch.tensor([self.x[idx]], dtype=torch.float32), 
#             'target': torch.tensor([self.y[idx]], dtype=torch.float32)
#         }
#         return sample
dataset = TensorDataset(x, y)
# Split dataset
train_set, test_set = random_split(dataset, [1-TEST_SIZE, TEST_SIZE])
x_train, x_test = x[train_set.indices], x[test_set.indices]
y_train, y_test = y[train_set.indices], y[test_set.indices]

###### 2. Define a model ######
# Define model
class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
    def forward(self, x):
        output = self.linear(x)
        return output
model = LR()
# Prediction after training
y_pred_before = model(x_test)
plt.plot(x_test, y_pred_before.detach(), label='prediction')
plt.scatter(x_test, y_test, label='data')
plt.legend()
plt.title('Prediction without training')
plt.show()

###### 3. Define Criterion & Optimizer ######
criterion = nn.MSELoss()  # Criterion (MSE)
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Optimizer (SGD)

###### 4. Training ######
losses = []  # Array for storing loss (criterion)
# Epoch loop (Without minibatch selection)
for epoch in range(NUM_EPOCHS):
    # Update parameters
    optimizer.zero_grad()  # Initialize gradient
    y_pred = model(x_train)  # Forward (Prediction)
    loss = criterion(y_pred, y_train)  # Calculate criterion
    loss.backward()  # Backpropagation (Calculate gradient)
    optimizer.step()  # Update parameters (Based on optimizer algorithm)
    if epoch % 10 ==0:  # Store and print loss every 10 epochs
        print(f'epoch: {epoch}, loss: {loss.item()}')
        losses.append(loss.item())

###### 5. Model evaluation and visualization ######
# Plot loss history
plt.plot(losses)
plt.title('Loss history')
plt.show()
# Prediction after training
y_pred_after = model(x_test)
plt.plot(x_test, y_pred_after.detach(), label='prediction')
plt.scatter(x_test, y_test, label='data')
plt.legend()
plt.title('Prediction after training')
plt.show()

# %%
