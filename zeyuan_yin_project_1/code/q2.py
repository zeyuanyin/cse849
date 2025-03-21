import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from layers import Linear, ReLU, MSE
import os

class MLP:
    def __init__(self, d_x, d_h, lr):
        self.lin1 = Linear(d_x, d_h, lr=lr)
        self.act1 = ReLU()
        self.lin2 = Linear(d_h, d_h, lr=lr)
        self.act2 = ReLU()
        self.lin3 = Linear(d_h, 1, lr=lr)

    def forward(self, X):
        a1 = self.lin1.forward(X)
        h1 = self.act1.forward(a1)
        a2 = self.lin2.forward(h1)
        h2 = self.act2.forward(a2)
        yhat = self.lin3.forward(h2)
        return yhat

    def backward(self, g):
        g = self.lin3.backward(g)
        g = self.act2.backward(g)
        g = self.lin2.backward(g)
        g = self.act1.backward(g)
        g = self.lin1.backward(g)

        return g

    def train(self):
        self.lin1.train()
        self.act1.train()
        self.lin2.train()
        self.act2.train()
        self.lin3.train()

    def eval(self):
        self.lin1.eval()
        self.act1.eval()
        self.lin2.eval()
        self.act2.eval()
        self.lin3.eval()


def main(lr = 1):
    mlp = MLP(2, 100, lr=lr)

    mse = MSE()
    mse.train()

    data = torch.load("Project1_data.pt", map_location="cpu",
                    weights_only=True)

    train_dataset = TensorDataset(data["x_train"], data["y_train"].reshape(-1, 1))
    val_dataset = TensorDataset(data["x_val"], data["y_val"].reshape(-1, 1))

    # print(len(train_dataset), len(val_dataset))
    # print(data["y_train"].max(), data["y_train"].min(), data["y_train"].mean())


    # # plot y values
    # plt.hist(data["y_train"], bins=100)
    # plt.savefig("q2_yhist.png", dpi=600, bbox_inches="tight")
    # # plt.show()
    # exit()

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_epochs = 100

    step = 0
    train_step_count = []
    val_step_count = []
    train_loss_list = []
    val_loss_list = []


    for i in trange(num_epochs):
        mlp.train()
        mse.train()
        
        acc = 0
        for batch in tqdm(train_dataloader, desc="Training", leave=False):
            X, y = batch
            # Run your forward pass
            yhat = mlp.forward(X)

            # Compute the loss
            loss = mse.forward(yhat, y)
            train_loss_list.append(loss.item())
            train_step_count.append(step)
            step += 1

            # Backpropagation
            g = mse.backward()
            g = mlp.backward(g)
            
            # calculate the accuracy
            acc += torch.sum(torch.abs(yhat - y) < 0.1).item()
        
        print(f"Epoch {i+1} \n Train Accuracy: {acc / len(data['y_train']) * 100:.2f}%, Loss: {loss.item():.4f}")
        
        mlp.eval()
        mse.eval()

        acc1 = 0
        # Compute the validation loss
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            X, y = batch
            yhat = mlp.forward(X)
            val_loss = mse.forward(yhat, y)
            val_loss_list.append(val_loss.item())
            val_step_count.append(step)
            
            acc1 += torch.sum(torch.abs(yhat - y) < 0.1).item()
        
        print(f"Validation Accuracy: {acc1 / len(data['y_val']) * 100:.2f}%, Loss: {val_loss.item():.4f}")

    mlp.eval()
    mse.eval()
    y_test = mlp.forward(data["x_test"]).squeeze()

    os.makedirs("plots_2", exist_ok=True)
    with open(f"q2_ytest_{lr}.txt", "w") as f:
        f.write("\n".join([str(x.item()) for x in y_test]))

    plt.plot(train_step_count, train_loss_list, label="Train Loss")
    plt.plot(val_step_count, val_loss_list, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE Loss")
    plt.title(f"Learning rate: {lr}")
    plt.ylim(0.001, 1)
    plt.legend()
    # plt.show()
    plt.savefig(f"plots_2/q2_{lr}.png", dpi=600, bbox_inches="tight")
    plt.clf()
    plt.close()
    
    return acc/len(data["y_train"]), acc1/len(data["y_val"])



if __name__ == "__main__":
    
    acc = []
    # for lr in [, ]:
    
    for lr in [10, 5, 1, 0.5, 0.1, 0.01, 0.001]:
        a,b = main(lr)
        acc.append((a,b))
    
    
    print(acc)