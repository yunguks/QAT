import torch
from typing import Any, Callable, Optional,Tuple, Union, Sequence
from itertools import repeat
import collections
from torchvision.utils import _log_api_usage_once
import warnings
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules import Module
import copy

__all__ = ["ConvNormActivation","Conv2dNormActivation","Quant_ReLU","Quant_ReLU6"]

def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )

class Quant_ReLU(Module):
    r"""Applies the rectified linear unit function element-wise:
    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    .. image:: ../scripts/activation_images/ReLU.png
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(Quant_ReLU, self).__init__()
        self.inplace = inplace
        self.relu = torch.nn.ReLU()

    def forward(self, input: Tensor) -> Tensor:

        for i in range(len(input)):
            temp = input[i].detach()
            M = torch.max(temp)
            m = torch.min(temp)
            input[i] = torch.round(254*(input[i]-m)/(M-m)-127)/1000
            input[i] = (1000*input[i]+127)*(M-m)/254+m
        
        return self.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
        
# class Quant_ReLU6(torch.nn.Module.activation.ReLU6):
class Quant_ReLU6(Module):
    r"""my Quant_ReLU6 Module.
    HardTanh is defined as:
    .. math::
        \text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}
    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``
    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    .. image:: ../scripts/activation_images/Hardtanh.png
    Examples::
        >>> m = nn.Hardtanh(-2, 2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['min_val', 'max_val', 'inplace']

    min_val: float
    max_val: float
    inplace: bool

    def __init__(
        self,
        min_val: float = 0.,   # hardtanh -1.
        max_val: float = 6.,    # hardtanh 1.
        inplace: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> None:
        super(Quant_ReLU6, self).__init__()
        if min_value is not None:
            warnings.warn("keyword argument min_value is deprecated and rename to min_val")
            min_val = min_value
        if max_value is not None:
            warnings.warn("keyword argument max_value is deprecated and rename to max_val")
            max_val = max_value

        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        assert self.max_val > self.min_val

    def forward(self, input: Tensor) -> Tensor:
        for i in range(input.size()[0]):
            M = torch.max(input[i])
            m = torch.min(input[i])
            input[i] = torch.round(254*(input[i]-m)/(M-m)-127)/1000
            input[i] = (1000*input[i]+127)*(M-m)/254+m
        # M = torch.max(torch.max(input),torch.full_like(input,0.127))
        # m = torch.min(torch.min(input),torch.full_like(input,-0.127))
        # input = torch.round(254*(input-m)/(M-m)-127) /1000 

        # input = (1000*input+127)*(M-m)/254+m
            
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'min_val={}, max_val={}{}'.format(
            self.min_val, self.max_val, inplace_str
        )
        
