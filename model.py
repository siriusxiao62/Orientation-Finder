import math
import torch
import torch.nn as nn

class GaborConv2dPC(nn.Module):
    def __init__(
        self,
        scale,
        orientation,
        in_channels=1,
        kernel_size=19,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
        pc=False,
        lam = 2,
    ):
        super().__init__()

        self.is_calculated = False

        self.pc = pc

        self.lam = lam

        out_channels = scale * orientation *2

        self.conv_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer.kernel_size

        # small addition to avoid division by zero
        self.delta = 1e-3

        torch.manual_seed(1)
        self.sigma = nn.Parameter(torch.rand(1), requires_grad=True)
        torch.manual_seed(1)
        self.sigma_list = torch.randn((out_channels, in_channels))
        torch.manual_seed(1)
        self.theta = nn.Parameter(torch.rand(1), requires_grad=True)
        torch.manual_seed(1)
        self.theta_list = torch.randn((out_channels, in_channels))
        for ori in range(orientation):  #8
            for s in range(scale): #6
                self.sigma_list[ori*scale+s] = self.sigma * (math.sqrt(2) ** s)
                self.theta_list[ori*scale+s] = self.theta + math.pi / 8 * ori

        torch.manual_seed(1)
        self.gamma = nn.Parameter(torch.rand(1), requires_grad=True)
        # print('gamma', self.gamma)
        torch.manual_seed(1)
        self.Lambda = nn.Parameter(torch.rand(1) +self.lam, requires_grad=True)
        print('Lambda', self.Lambda)
        torch.manual_seed(1)
        self.psi = nn.Parameter(torch.rand(1), requires_grad=True)


        self.x0 = nn.Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False
        )
        self.y0 = nn.Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ]
        )
        self.y = nn.Parameter(self.y)
        self.x = nn.Parameter(self.x)

        self.weight = nn.Parameter(
            torch.empty(self.conv_layer.weight.shape, requires_grad=True),
            requires_grad=True,
        )
        torch.manual_seed(1)
        self.coeff = nn.Parameter(torch.rand(1, scale * orientation, 1), requires_grad=True)
        # print('coeff', self.coeff)
        torch.manual_seed(1)
        self.w = nn.Parameter(torch.rand(1), requires_grad=True)
        # print('w', self.w)
        torch.manual_seed(1)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)
        # print('b', self.b)

        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("Lambda", self.Lambda)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)
        self.register_parameter("coeff", self.coeff)
        self.register_parameter("w", self.w)
        self.register_parameter("b", self.b)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        output = self.conv_layer(input_tensor)

        if not self.pc:
            return output
#         print(output.shape)

        else:
            x_cos = output[:,:self.conv_layer.out_channels//2,:,:]
            x_sin = output[:,self.conv_layer.out_channels//2:,:,:]
            x_comb = torch.stack((x_cos, x_sin), 4)  #5 dim tensor
            # x_cos = x_cos.view(len(input_tensor), x_cos.shape[2], x_cos.shape[3], 48)
            x_cos = x_cos.permute(0,2,3,1)
            # x_sin = x_sin.view(len(input_tensor), x_sin.shape[2], x_sin.shape[3], 48)
            x_sin = x_sin.permute(0, 2, 3, 1)
            weighted_cos = (torch.matmul(x_cos, self.coeff))#.view(len(input_tensor), x_cos.shape[1], x_cos.shape[2])
            weighted_sin = (torch.matmul(x_sin, self.coeff))#.view(len(input_tensor), x_sin.shape[1], x_sin.shape[2])
            numerator = torch.norm(torch.stack([weighted_cos, weighted_sin], 4), dim=4)  #(batch, 501, 501,1)
            x_comb_norm = torch.norm(x_comb, dim=4) #back to 4 dim, 1*48*501*501
            # x_comb_norm = x_comb_norm.view(len(input_tensor),x_comb_norm.shape[2], x_comb_norm.shape[3], 48)
            x_comb_norm = x_comb_norm.permute(0, 2, 3, 1)
            denominator = torch.matmul(x_comb_norm, torch.abs(self.coeff))  ##(batch, 501, 501,1)
            # denominator = denominator.view(len(input_tensor))
            pc = numerator / denominator

            return pc

    def calculate_weights(self):
        for i in range(self.conv_layer.out_channels//2):
            for j in range(self.conv_layer.in_channels):

                rotx = self.x * torch.cos(self.theta_list[i, j]) + self.y * torch.sin(self.theta_list[i, j])
                roty = -self.x * torch.sin(self.theta_list[i, j]) + self.y * torch.cos(self.theta_list[i, j])

                g = torch.exp(
                    -0.5 * ((rotx ** 2 + (self.gamma **2) * (roty ** 2)) / (self.sigma_list[i, j]**2 + self.delta))
                )
                g_cos = g * torch.cos(2 * math.pi * rotx / self.Lambda + self.psi)
                g_sin = g * torch.sin(2 * math.pi * rotx / self.Lambda + self.psi)

                self.conv_layer.weight.data[i, j] = g_cos
                self.conv_layer.weight.data[i+self.conv_layer.out_channels//2, j] = g_sin

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# 7x7 convolution
def conv7x7(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7,
                     stride=stride, padding=3, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, channels, input_channel):
        super(ResNet, self).__init__()
        self.in_channels = channels[0]
        self.conv = conv3x3(input_channel, self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, self.in_channels, layers[0])
        self.layer2 = self.make_layer(block, channels[1], layers[1])
        self.layer3 = self.make_layer(block, channels[2], layers[2])
        self.layer4 = self.make_layer(block, channels[3], layers[3])
        self.output1 = conv3x3(channels[3], channels[1])
        self.output_bn = nn.BatchNorm2d(channels[1])
        self.output = conv3x3(channels[1], 1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.output_bn(self.output1(out))
        out = nn.Sigmoid()(self.output(out))
        return out

class ResNet_gabor_cat2(nn.Module):  #concat two gabor
    def __init__(self, block, layers, channels, scale, orientation, kernel_size, pc = False, lam = [2, 20]):
        super(ResNet_gabor_cat2, self).__init__()
        self.gabor1 = GaborConv2dPC(scale=scale, orientation= orientation, kernel_size=kernel_size,
                                    padding=kernel_size//2, pc=pc, lam=lam[0])
        self.gabor2 = GaborConv2dPC(scale=scale, orientation=orientation, kernel_size=kernel_size,
                                    padding=kernel_size // 2, pc=pc, lam=lam[1])
        if pc:
            input_channel = 1    #channel of the input image
        else:
            input_channel = 2 * scale * orientation  #channel of the gabor filter output if PC is not calculated
        self.resnet = ResNet(block, layers, channels, input_channel + len(lam)-1)

    def forward(self, x):
        out1 = self.gabor1(x)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = self.gabor2(x)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = torch.cat([out1, out2], dim = 1)
        # print('cat', out2.shape)
        out = self.resnet(out3)
        return out
































