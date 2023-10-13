import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to PyTorch tensors
        state = np.array(state)
        next_state = np.array(next_state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # If the input is a single sample, add a batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # Predicted Q values with the current state
        pred = self.model(state)

        # Clone the predictions to create the target
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the Q value for the action taken
            target[idx][action[idx]] = Q_new

        # Zero the gradients, calculate the loss, and perform backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# Example usage
if __name__ == '__main__':
    input_size = 3  # Replace with your input size
    hidden_size = 256  # Replace with your desired hidden size
    output_size = 2  # Replace with your output size
    model = Linear_QNet(input_size, hidden_size, output_size)
    trainer = QTrainer(model, lr=0.001, gamma=0.9)

    # Sample data (replace with your own data)
    state = [0.1, 0.2, 0.3]
    action = 0
    reward = 1.0
    next_state = [0.2, 0.3, 0.4]
    done = False

    # Train the model with a single step
    trainer.train_step(state, action, reward, next_state, done)
