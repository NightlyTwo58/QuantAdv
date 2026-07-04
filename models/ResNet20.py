import torch

def ResNet3x3Conv(in_channels: int, out_channels: int, stride=1) -> torch.nn.Conv2d:
    """
    3x3 convolution
    """
    return torch.nn.Conv2d(in_channels=in_channels, 
                           out_channels=out_channels, 
                           kernel_size=3,
                           stride=stride,
                           padding=1)

def ResNet1x1Conv(in_channels: int, out_channels: int) -> torch.nn.Conv2d:
    """
    1x1 convolution that doubles the amount of channels in the shape of the tensor.
    Used for "widening" blocks that double the amount of filters.
    """
    return torch.nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           stride=1)

class ResNetStandardBlock(torch.nn.Module):
    def __init__(self, dim: int, in_channels: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            ResNet3x3Conv(dim, dim),
            torch.nn.ReLU(),
            ResNet3x3Conv(dim, dim)
        )
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        # residual behavior
        return self.relu(self.layers.forward(x) + x)

class ResNetWideningBlock(torch.nn.Module):
    def __init__(self, dim: int, in_channels: int):
        super().__init__
        self.layers = torch.nn.Sequential(
            ResNet3x3Conv(dim, dim * 2, 2),
            torch.nn.ReLU(),
            ResNet3x3Conv(dim * 2, dim * 2)
        )
        self.residual = ResNet1x1Conv(dim, dim * 2)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.layers.forward(x) + self.residual.forward(x))

def _initialize_structure(blocks: int) -> torch.nn.Sequential:
    model = torch.nn.Sequential()
    model.append(ResNet3x3Conv(3, 16))
    model.append(torch.nn.ReLU())

    for i in range(1, blocks + 1):
        model.append(ResNetStandardBlock(dims=16 * blocks, in_channels=16))
        model.append(ResNetWideningBlock(dims=16 * blocks, in_channels=16))

    model.append(torch.nn.AvgPool2d(8))
    model.append(torch.nn.Linear(64, 10))
    model.append(torch.nn.Softmax(10))
    