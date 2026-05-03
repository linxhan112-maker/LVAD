import torch
import torch.nn as nn


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, l = x.size()

        n = l - 1

        x_minus_mu_square = (x - x.mean(dim=2, keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=2, keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class crosssimam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(crosssimam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "crosssimam"

    def forward(self, x1, x2):
        b, c, l = x1.size()

        n = l - 1

        x_minus_mu_square = (x1 - x1.mean(dim=2, keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=2, keepdim=True) / n + self.e_lambda)) + 0.5

        y = 1 / (x2.pow(2) + self.e_lambda) * y

        return x1 * self.activaton(y)


class LowLightSIMAM(nn.Module):
    def __init__(self, e_lambda=1e-4, kernel_size=3):
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()
        self.local_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        B, C, H, W = x.size()
        local_illumination = self.local_pool(x)
        mu = x.mean(dim=(2, 3), keepdim=True)
        x_var = (x - mu).pow(2).mean(dim=(2, 3), keepdim=True)
        # 修正：分母改为除以局部光照
        #denominator = 4 * (x_var + self.e_lambda) / (local_illumination + 0.1)
        denominator = 4 * (x_var + self.e_lambda) * (local_illumination + 0.1) + 1e-6
        y = (x - mu).pow(2) / denominator + 0.5
        return x * self.activation(y)