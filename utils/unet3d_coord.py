import torch
import torch.nn as nn


class CoordConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, kernel_size=3, stride=1, padding=1):
        super(CoordConv3d, self).__init__()
        self.with_r = with_r
        self.conv = nn.Conv3d(in_channels + 3 + int(with_r), out_channels, kernel_size, stride, padding, padding_mode='replicate')

    def forward(self, x):
        # get shape information of the input data
        batch_size, _, z_dim, y_dim, x_dim = x.size()

        # generate the coordinate channels for x
        xx_range = torch.linspace(0, x_dim - 1, x_dim)
        xx_channel = xx_range.unsqueeze(0).unsqueeze(0).repeat(z_dim, y_dim, 1)
        xx_channel = xx_channel.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        # generate the coordinate channels for y
        yy_range = torch.linspace(0, y_dim - 1, y_dim)
        yy_channel = yy_range.unsqueeze(0).unsqueeze(2).repeat(z_dim, 1, x_dim)
        yy_channel = yy_channel.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        # generate the coordinate channels for z
        zz_range = torch.linspace(0, z_dim - 1, z_dim)
        zz_channel = zz_range.unsqueeze(1).unsqueeze(2).repeat(1, y_dim, x_dim)
        zz_channel = zz_channel.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        # To device
        device = x.device
        xx_channel = xx_channel.to(device)
        yy_channel = yy_channel.to(device)
        zz_channel = zz_channel.to(device)

        # normalized coordinates to [0, 1]
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        zz_channel = zz_channel.float() / (z_dim - 1)

        # adjust the coordinate range to [-1, 1]
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        zz_channel = zz_channel * 2 - 1

        # concatenate input data with x,y, z coordinates
        x = torch.cat([x, xx_channel, yy_channel, zz_channel], dim=1)

        # add radius channel if radius is True
        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_channel ** 2 + zz_channel ** 2)
            x = torch.cat([x, rr], dim=1)

        x = self.conv(x)
        return x


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        nn.InstanceNorm3d(out_dim),
        activation
    )


def conv_block_coord_3d(in_dim, out_dim, activation, with_r):
    return nn.Sequential(
        CoordConv3d(in_dim, out_dim, with_r, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation
    )


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        conv_block_3d(out_dim, out_dim, activation)
    )


def conv_block_coord_2_3d(in_dim, out_dim, activation, with_r):
    return nn.Sequential(
        conv_block_coord_3d(in_dim, out_dim, activation, with_r),
        conv_block_3d(out_dim, out_dim, activation)
    )


def up_conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        nn.InstanceNorm3d(out_dim),
        activation
    )


def conv_block_out_3d(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
    )


def conv_block_out_activate_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        activation
    )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


class UnetL4(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters, activation=nn.LeakyReLU(0.01, inplace=True), output_activation=None, with_r=False):
        super(UnetL4, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        self.activation = activation
        self.output_activation = output_activation
        self.with_r = with_r

        # Down sampling
        self.down_1 = conv_block_coord_2_3d(self.in_dim, self.num_filters, self.activation, self.with_r)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, self.activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, self.activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, self.activation)
        self.pool_4 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, self.activation)

        # Up sampling
        self.trans_1 = up_conv_block_3d(self.num_filters * 16, self.num_filters * 16, self.activation)
        self.up_1 = conv_block_2_3d(self.num_filters * (16 + 8), self.num_filters * 8, self.activation)
        self.trans_2 = up_conv_block_3d(self.num_filters * 8, self.num_filters * 8, self.activation)
        self.up_2 = conv_block_2_3d(self.num_filters * (8 + 4), self.num_filters * 4, self.activation)
        self.trans_3 = up_conv_block_3d(self.num_filters * 4, self.num_filters * 4, self.activation)
        self.up_3 = conv_block_2_3d(self.num_filters * (4 + 2), self.num_filters * 2, self.activation)
        self.trans_4 = up_conv_block_3d(self.num_filters * 2, self.num_filters * 2, self.activation)
        self.up_4 = conv_block_2_3d(self.num_filters * (2 + 1), self.num_filters, self.activation)

        # Output without activation
        if self.output_activation:
            self.out = conv_block_out_activate_3d(self.num_filters, self.out_dim, self.output_activation)
        else:
            self.out = conv_block_out_3d(self.num_filters, self.out_dim)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        # Bridge
        bridge = self.bridge(pool_4)

        # Up sampling
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.up_1(concat_1)

        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.up_2(concat_2)

        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.up_4(concat_4)

        # Output
        out = self.out(up_4)
        return out

