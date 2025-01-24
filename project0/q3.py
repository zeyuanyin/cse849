import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# seeds_list = [1, 2, 3, 4, 5]
seeds_list = [1]
# TODO: For each seed, plot the fitted model along with the training
# data points. The data samples need to be plotted only once. Follow the
# instructions from q2.py on how to plot the results.
y_hat_list = []
for seed in seeds_list:
    torch.manual_seed(seed)

    """
    Load the dataset from zip files
    """
    data = torch.load("HW0_data.pt", weights_only=True)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_test = data["x_test"]
    """
    Create corresponding datasets.
    """
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(x_test)  # skip y_test in the test dataset

    """
    Create dataloaders for each dataset.
    """
    batch_size = 2000  # Use the entire dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    """
    Create the MLP as described in the PDF
    """
    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.LeakyReLU(),
        nn.Linear(10, 10),
        nn.LeakyReLU(),
        nn.Linear(10, 1)
    ).cuda()

    """
    Initialize the model after setting PyTorch seed to model_seed. But we
    also don't want to disturb other parts of the code. Therefore, we will
    store the current random state, set the seed to model_seed, initialize
    the model and restore the random state.
    """
    rng = torch.get_rng_state()
    # TODO: set seed
    model_seed = 90
    torch.manual_seed(model_seed)
    # TODO: Sample the values from a normal distribution
    for layer in model:
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)

    torch.set_rng_state(rng)

    """
    Create an AdamW optimizer with the given learning rate lr and
    weight_decay. Check the usage of the optimizer in the PyTorch documentation.
    Make sure to pass the parameters of model to the optimizer.
    """
    lr = 1e-2
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    """
    Set up the loss function as nn.MSELoss.
    """
    loss_fn = nn.MSELoss()

    num_epochs = 1000

    train_step_list = []
    train_loss_list = []
    val_step_list = []
    val_loss_list = []
    w_list = []
    step = 0

    for e in trange(num_epochs):
        # TODO: Set your model to training mode.
        model.train()
        for batch in tqdm(train_loader, leave=False, desc="Training"):
            # TODO: Zero the gradients
            optimizer.zero_grad()

            # TODO: Unpack the batch. It is a tuple containing x and y.
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)

            # TODO: Pass it through the model to get the predicted y_hat.
            y_hat = model(x)

            # TODO: Calculate the loss using the loss function.
            loss = loss_fn(y_hat, y)

            # TODO: Backpropagate the loss.
            loss.backward()

            # TODO: Update the model weights.
            optimizer.step()

            # TODO: Store the training loss in the list
            train_step_list.append(step)
            train_loss_list.append(loss.item())
            step += 1


    # plot the training data and predicted labels
    model.eval()
    train_loader_plot = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # no shuffle to plot the data in order
    y_hat_list_seed = []
    with torch.no_grad():
        for batch in train_loader_plot:
            x, y = batch
            x = x.cuda()
            x = x.unsqueeze(-1)
            y_hat = model(x)
            y_hat_list_seed.extend(y_hat.cpu().numpy().flatten())
    y_hat_list.append(y_hat_list_seed)



    # TODO: Evaluate your model on the validation set.
    # Remember to set the model in evaluation mode and to use
    # torch.no_grad()



# Plot the results
plt.scatter(x_train, y_train, c="black", marker="^", label="Real Data")
colors = ["blue", "green", "red", "yellow", "purple"]
for i, y_hat in enumerate(y_hat_list):
    plt.scatter(x_train, y_hat, c=colors[i], label=f"Seed {seeds_list[i]}")
plt.grid(True)
plt.legend()
plt.savefig("q3_plot.png")

# # TODO: Run the model on the test set
# with torch.no_grad():
#     yhat_test = None

# with open("q3_test_output.txt", "w") as f:
#     for yhat in yhat_test:
#         f.write(f"{yhat.item()}\n")

# # TODO: Save the model as q3_model.pt
# torch.save(model.state_dict(), f'q3_model.pt')
