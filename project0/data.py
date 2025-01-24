import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, alpha, beta, num_samples,
                 split, seed):
        super().__init__()
        """
        TODO: Set the seed according to the split. If the split is train, use
        the seed from the arguments, and if the split is val, use the
        seed from the arguments + 1.
        """
        if split == 'train':
            torch.manual_seed(seed)
        else:
            torch.manual_seed(seed + 1)

        """
        TODO: Generate the input x using torch.randn containing num_samples
        samples
        """
        self.x = torch.randn(num_samples)

        """
        TODO: Generate a noise vector e using torch.randn containing
        num_samples samples
        """
        e = torch.randn(num_samples)


        """
        TODO: From x and e, generate the expected output y according to the
        expression:
        y = alpha * x + beta + e
        """
        self.y = alpha * self.x + beta + e

    def __len__(self):
        """
        TODO: Return the number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, index):
        """
        TODO: Return the values of x and y indexed by the argument
        index. Remember to reshape the values to be of shape (1,).
        """
        return self.x[index].reshape(1,), self.y[index].reshape(1,)




if __name__ == '__main__':
    dataset = MyDataset(alpha=1, beta=0, num_samples=10, split='train', seed=0)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][0])
    print(dataset[0][1])
    exit()
    import matplotlib.pyplot as plt
    plt.scatter(dataset.x, dataset.y, c='blue', marker='.')
    plt.grid(True)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.savefig('dataset.png')

    x, y = dataset[0:10]
    
    print(x)
    print(y)
    exit()
