import torch
import torch.nn as nn


class SleepApneaCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # ---------------------------------------
        # CONVOLUTION BLOCK 1
        # ---------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=35,
            stride=1,
            padding=17
        )

        self.bn1 = nn.BatchNorm1d(
            8
        )

        self.elu1 = nn.ELU()

        self.pool1 = nn.MaxPool1d(
            kernel_size=7,
            stride=7
        )

        self.dropout1 = nn.Dropout(
            p=0.1
        )

        # ---------------------------------------
        # CONVOLUTION BLOCK 2
        # ---------------------------------------

        self.conv2 = nn.Conv1d(
            in_channels=8,
            out_channels=128,
            kernel_size=175,
            stride=1,
            padding=87
        )

        self.bn2 = nn.BatchNorm1d(
            128
        )

        self.elu2 = nn.ELU()

        self.pool2 = nn.MaxPool1d(
            kernel_size=7,
            stride=7
        )

        self.dropout2 = nn.Dropout(
            p=0.1
        )

        # ---------------------------------------
        # CONVOLUTION BLOCK 3
        # ---------------------------------------

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=16,
            kernel_size=175,
            stride=1,
            padding=87
        )

        self.bn3 = nn.BatchNorm1d(
            16
        )

        self.elu3 = nn.ELU()

        self.pool3 = nn.MaxPool1d(
            kernel_size=7,
            stride=7
        )

        self.dropout3 = nn.Dropout(
            p=0.1
        )

        # ---------------------------------------
        # CLASSIFIER
        # ---------------------------------------

        self.flatten = nn.Flatten()

        # Input length:
        # 3750
        # -> pool 7 = 535
        # -> pool 7 = 76
        # -> pool 7 = 10
        #
        # 16 channels * 10 = 160

        self.fc1 = nn.Linear(
            160,
            64
        )

        self.elu4 = nn.ELU()

        self.output = nn.Linear(
            64,
            2
        )

        self.initialize_weights()

    def initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv1d
            ):

                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=0.05
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.Linear
            ):

                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=0.05
                )

                nn.init.zeros_(
                    module.bias
                )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.elu4(x)

        logits = self.output(x)

        return logits


if __name__ == "__main__":

    model = SleepApneaCNN()

    test_input = torch.randn(
        8,
        1,
        3750
    )

    output = model(
        test_input
    )

    print(model)
    print()
    print(
        "Input shape:",
        test_input.shape
    )

    print(
        "Output shape:",
        output.shape
    )