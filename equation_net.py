import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import PolynomialLR
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import wandb

# CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 생성
def generate_data(num_samples=1000):
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    y1 = x**2 + 2*y
    y2 = -x + y**2
    return torch.tensor(x, dtype=torch.float32).reshape(-1, 1), torch.tensor(y, dtype=torch.float32).reshape(-1, 1), torch.tensor(y1, dtype=torch.float32).reshape(-1, 1), torch.tensor(y2, dtype=torch.float32).reshape(-1, 1)

# 모델 정의
class EquationNet(nn.Module):
    def __init__(self, layers):
        super(EquationNet, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(layers) - 1):
            self.model.add_module(f"fc{i+1}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # No activation for the output layer
                self.model.add_module(f"gelu{i+1}", nn.GELU())
    
    def forward(self, x):
        return self.model(x)

# 데이터 준비
x, y, y1, y2 = generate_data()
inputs = torch.cat((x, y), dim=1)
targets = torch.cat((y1, y2), dim=1)
dataset = TensorDataset(inputs, targets)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델, 손실 함수, 옵티마이저, 스케줄러 설정
layers = [2, 100, 100, 100, 2]
model = EquationNet(layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = PolynomialLR(optimizer, total_iters=3000, power=2)

# 모델 파일 경로
model_path = "equation_net.pth"

# 모델 학습
if not os.path.exists(model_path):
    # wandb 초기화
    wandb.init(project="EquationNet")
    wandb.watch(model, log="all")
    num_epochs = 3000
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "val_loss": val_loss
        })
    
    # 모델 저장
    torch.save(model.state_dict(), model_path)
    wandb.finish()
else:
    # 모델 불러오기
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

# 학습이 끝난 후 플로팅
def plot_function_and_derivative(fixed_var, var_values, var_name, save_path):
    outputs = []
    grads = []

    for value in var_values:
        if var_name == 'x':
            inputs = torch.tensor([[value, fixed_var]], dtype=torch.float32).to(device).detach().clone().requires_grad_(True)
        else:
            inputs = torch.tensor([[fixed_var, value]], dtype=torch.float32).to(device).detach().clone().requires_grad_(True)

        output = model(inputs).squeeze(0)
        outputs.append(output.detach().cpu().numpy())

        if var_name == 'x':
            output[0].backward(retain_graph=True)
        else:
            output[1].backward(retain_graph=True)

        grads.append(inputs.grad[0].detach().cpu().numpy().copy())
        inputs.grad.zero_()

    outputs = np.array(outputs)
    grads = np.array(grads)

    with plt.style.context(['science', 'nature']):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(var_values, outputs[:, 0], label='y1')
        plt.plot(var_values, outputs[:, 1], label='y2')
        plt.title(f'Output with {var_name}=fixed_var')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(var_values, grads[:, 0], label='dy1/d'+var_name)
        plt.plot(var_values, grads[:, 1], label='dy2/d'+var_name)
        plt.title(f'Derivative with {var_name}=fixed_var')
        plt.legend()

        plt.savefig(save_path, dpi=600)

# y=1 고정
plot_function_and_derivative(1, np.linspace(-1, 1, 100, dtype=np.float32), 'x', 'output_with_y_fixed.png')

# x=1 고정
plot_function_and_derivative(1, np.linspace(-1, 1, 100, dtype=np.float32), 'y', 'output_with_x_fixed.png')
