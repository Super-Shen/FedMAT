from turtle import forward
import torch
import torch.nn as nn



class my_cnn(nn.Module):
    def __init__(self):
        super(my_cnn, self).__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 2)),
            nn.BatchNorm3d(64),  # .1.weight / .1.bias / .1.running_mean / .1.running_ver / .1.num_batches_tracked
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 3), padding=(0, 0)),
            # nn.BatchNorm3d(output_channels),
            # nn.ReLU(),
            # nn.Dropout(p=zero_prob),
        )
    
    def forward(self, x):
        pass

class my_cnn_2(nn.Module):
    def __init__(self):
        super(my_cnn_2, self).__init__()

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 2)),
            nn.BatchNorm3d(64),  # .1.weight / .1.bias / .1.running_mean / .1.running_ver / .1.num_batches_tracked
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 3), padding=(0, 0)),
            # nn.BatchNorm3d(output_channels),
            # nn.ReLU(),
            # nn.Dropout(p=zero_prob),
        )
    
    def forward(self, x):
        pass


if __name__ == "__main__":
    cnn = my_cnn()
    cnn_conv3d_w, cnn_conv2d_w = cnn.conv3d.state_dict(), cnn.conv2d.state_dict()

    cnn_2 = my_cnn_2()
    cnn_2.conv3d_1.load_state_dict(cnn_conv3d_w)
    cnn_2.conv2d_1.load_state_dict(cnn_conv2d_w)
    print()