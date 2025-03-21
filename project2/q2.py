import torch
import imageio.v2 as imio
import os

from model import CNN

model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the weights
model.load_state_dict(torch.load("q1_plots_W_bs32_lr0.005_Erasing_onecycle_0.3_25_10000_E300.pt", weights_only=True))

model.eval()

conv_weights = model.conv1.weight.data.cpu().numpy()  # Get the conv1 layer weights

os.makedirs("q2_filters", exist_ok=True)

# print(conv_weights.shape) # (16, 3, 7, 7)

for i in range(conv_weights.shape[0]):
    f = conv_weights[i, 0, :, :]  # get the first channel of the i-th filter
    f_min, f_max = f.min(), f.max()
    f = (f - f_min) / (f_max - f_min) * 255
    f = f.astype('uint8')
    
    # Otherwise, it will not be visualized correctly.
    imio.imwrite(f"q2_filters/filter_{i}.png", f)
