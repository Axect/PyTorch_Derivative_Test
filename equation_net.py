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

# Set CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate data
def generate_data(num_samples=1000):
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    y1 = x**2 + 2*y
    y2 = -x + y**2
    return torch.tensor(x, dtype=torch.float32).reshape(-1, 1), torch.tensor(y, dtype=torch.float32).reshape(-1, 1), torch.tensor(y1, dtype=torch.float32).reshape(-1, 1), torch.tensor(y2, dtype=torch.float32).reshape(-1, 1)

# Define model
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

# Prepare data
x, y, y1, y2 = generate_data()
inputs = torch.cat((x, y), dim=1)
targets = torch.cat((y1, y2), dim=1)
dataset = TensorDataset(inputs, targets)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Set model, loss function, optimizer, scheduler
layers = [2, 100, 100, 100, 2]
model = EquationNet(layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = PolynomialLR(optimizer, total_iters=3000, power=2)

# Model file path
model_path = "equation_net.pth"

# Train model
if not os.path.exists(model_path):
    # Initialize wandb
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
    
    # Save model
    torch.save(model.state_dict(), model_path)
    wandb.finish()
else:
    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

# Plot after training
def plot_function_and_derivative(fixed_var, var_values, var_name, save_path):
    outputs = []
    grads_y1 = []
    grads_y2 = []

    for value in var_values:
        if var_name == 'x':
            inputs = torch.tensor([[value, fixed_var]], dtype=torch.float32).to(device).detach().clone().requires_grad_(True)
        else:
            inputs = torch.tensor([[fixed_var, value]], dtype=torch.float32).to(device).detach().clone().requires_grad_(True)

        output = model(inputs).squeeze(0)
        outputs.append(output.detach().cpu().numpy())

        # dy1/dx, dy2/dy
        output[0].backward(retain_graph=True)

        if var_name == 'x':
            grads_y1.append(inputs.grad[0][0].detach().cpu().numpy().copy())
        else:
            grads_y1.append(inputs.grad[0][1].detach().cpu().numpy().copy())
        inputs.grad.zero_()

        # dy2/dx, dy2/dy
        output[1].backward(retain_graph=True)

        if var_name == 'x':
            grads_y2.append(inputs.grad[0][0].detach().cpu().numpy().copy())
        else:
            grads_y2.append(inputs.grad[0][1].detach().cpu().numpy().copy())
        inputs.grad.zero_()

    outputs = np.array(outputs)
    grads_y1 = np.array(grads_y1)
    grads_y2 = np.array(grads_y2)

    other_var = 'x' if var_name == 'y' else 'y'

    true_val_y1 = []
    true_val_y2 = []
    true_grad_y1 = []
    true_grad_y2 = []

    if var_name == 'x':
        true_val_y1 = var_values ** 2
        true_val_y2 = -var_values
        true_grad_y1 = 2 * var_values
        true_grad_y2 = -np.ones_like(var_values)
    else:
        true_val_y1 = 2 * var_values
        true_val_y2 = var_values ** 2
        true_grad_y1 = np.ones_like(var_values) * 2
        true_grad_y2 = 2 * var_values

    with plt.style.context(['science', 'nature']):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(var_values, outputs[:, 0], color='darkblue', label=r'$\hat{y}_1$')
        plt.plot(var_values, true_val_y1, ':', color='red', label=r'$y_1$')
        plt.plot(var_values, outputs[:, 1], color='darkgreen', label=r'$\hat{y}_2$')
        plt.plot(var_values, true_val_y2, ':', color='orange', label=r'$y_2$')
        plt.title(f'Output with {other_var}=0')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(var_values, grads_y1[:], color='darkblue', label=r'$d\hat{y_1}/d'+var_name+r'$')
        plt.plot(var_values, true_grad_y1, ':', color='red', label=r'$dy_1/d'+var_name+r'$')
        plt.plot(var_values, grads_y2[:], color='darkgreen', label=r'$d\hat{y_2}/d'+var_name+r'$')
        plt.plot(var_values, true_grad_y2, ':', color='orange', label=r'$dy_2/d'+var_name+r'$')
        plt.title(f'Derivative with {other_var}=0')
        plt.legend()

        plt.savefig(save_path, dpi=600)

# Fixed y=1
plot_function_and_derivative(0, np.linspace(-1, 1, 100, dtype=np.float32), 'x', 'output_x.png')

# Fixed x=1
plot_function_and_derivative(0, np.linspace(-1, 1, 100, dtype=np.float32), 'y', 'output_y.png')
