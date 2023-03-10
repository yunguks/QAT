from functools import partial
from typing import Any, Callable, List, Optional, Dict, TypeVar, Mapping, Tuple

import torch
from torch import nn, Tensor

from .layers import Conv2dNormActivation, Quant_ReLU, Quant_ReLU6
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
from torchvision._utils import StrEnum
from typing import Any, List, Optional, Union
import warnings

__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "mobilenet_v2",
           "QuantizableMobileNetV2","quat_mobilenet_v2",
           "quantize_model","QuantizableInvertedResidual",
           "replace_relu", "replace_Qrelu"]

# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer :Optional[Callable[..., nn.Module]]= nn.ReLU6
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1
        
    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        activation_layer= nn.ReLU6,
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()
        _log_api_usage_once(self)

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=activation_layer)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer,activation_layer=activation_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


_COMMON_META = {
    "num_params": 3504872,
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class MobileNet_V2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.878,
                    "acc@5": 90.286,
                }
            },
            "_ops": 0.301,
            "_file_size": 13.555,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.154,
                    "acc@5": 90.822,
                }
            },
            "_ops": 0.301,
            "_file_size": 13.598,
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@handle_legacy_interface(weights=("pretrained", MobileNet_V2_Weights.IMAGENET1K_V1))
def mobilenet_v2(
    *, weights: Optional[MobileNet_V2_Weights] = None,
    progress: bool = True,
    cifar10:bool = False,
    **kwargs: Any
) -> MobileNetV2:
    """MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and Linear
    Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.
    Args:
        weights (:class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenetv2.MobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
    """
    weights = MobileNet_V2_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    
    if cifar10:
        inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                # [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
                [4, 32, 3, 2],
                [4, 64, 4, 2],
                # [6, 96, 3, 1],
                [4, 128, 3, 2],
                # [6, 320, 1, 1],
            ]
        kwargs["num_classes"]=10
        kwargs['inverted_residual_setting']=inverted_residual_setting

    model = MobileNetV2(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


# The dictionary below is internal implementation detail and will be removed in v0.15
from torchvision.models._utils import _ModelURLs


model_urls = _ModelURLs(
    {
        "mobilenet_v2": MobileNet_V2_Weights.IMAGENET1K_V1.url,
    }
)

class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) is nn.Conv2d:
                _fuse_modules(self.conv, [str(idx), str(idx + 1)], is_qat, inplace=True)


class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V2 main class
        Args:
           Inherits args from floating point MobileNetV2
        """
        super().__init__(*args, **kwargs)
        _log_api_usage_once(self)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for m in self.modules():
            if type(m) is Conv2dNormActivation:
                _fuse_modules(m, ["0", "1", "2"], is_qat, inplace=True)
            if type(m) is QuantizableInvertedResidual:
                m.fuse_model(is_qat)

def _fuse_modules(
    model: nn.Module, modules_to_fuse: Union[List[str], List[List[str]]], is_qat: Optional[bool], **kwargs: Any
):
    if is_qat is None:
        is_qat = model.training
    method = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
    return method(model, modules_to_fuse, **kwargs)

class MobileNet_V2_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            "num_params": 3504872,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "backend": "qnnpack",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv2",
            "unquantized": MobileNet_V2_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.658,
                    "acc@5": 90.150,
                }
            },
            "_ops": 0.301,
            "_file_size": 3.423,
            "_docs": """
                These weights were produced by doing Quantization Aware Training (eager mode) on top of the unquantized
                weights listed below.
            """,
        },
    )
    DEFAULT = IMAGENET1K_QNNPACK_V1
    
def quat_mobilenet_v2(
    *,
    weights: Optional[Union[MobileNet_V2_QuantizedWeights, MobileNet_V2_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    cifar10: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks
    <https://arxiv.org/abs/1801.04381>`_.
    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.
    Args:
        weights (:class:`~torchvision.models.quantization.MobileNet_V2_QuantizedWeights` or :class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.MobileNet_V2_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        quantize (bool, optional): If True, returns a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableMobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv2.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.quantization.MobileNet_V2_QuantizedWeights
        :members:
    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
        :noindex:
    """
    weights = (MobileNet_V2_QuantizedWeights if quantize else MobileNet_V2_Weights).verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
    backend = kwargs.pop("backend", "qnnpack")

    if cifar10:
        inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                # [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
                [4, 32, 3, 2],
                [4, 64, 4, 2],
                # [6, 96, 3, 1],
                [4, 128, 3, 2],
                # [6, 320, 1, 1],
            ]
        kwargs["num_classes"]=10
        kwargs['inverted_residual_setting']=inverted_residual_setting

    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)
    
    if quantize:
        replace_relu(model)
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        
    
    return model

def replace_relu(module: nn.Module) -> None:
    reassign = {}
    for name, mod in module.named_children():
        replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6 or type(mod) is Quant_ReLU or type(mod) is Quant_ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value
        
def replace_Qrelu(module: nn.Module) -> None:
    reassign = {}
    for name, mod in module.named_children():
        replace_Qrelu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6:
            reassign[name] = Quant_ReLU6(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value

def quantize_model(model: nn.Module,data, backend: str = "fbgemm", qconfig = None) -> None:
    """_summary_

    Args:
        model (nn.Module): 
        data (data): calibrate
        backend (str, optional):  Defaults to "fbgemm".
        qconfig ( ): Defaults to None

    Raises:
        RuntimeError: _description_
    """
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    replace_relu(model)
    model.eval()
    model.fuse_model()  # type: ignore[operator]
    
    # Make sure that weight qconfig matches that of the serialized models
    if qconfig is None:
        if backend == "fbgemm":
            model.qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
                activation=torch.ao.quantization.default_observer,
                weight=torch.ao.quantization.default_per_channel_weight_observer,
            )
        elif backend == "qnnpack":
            model.qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
                activation=torch.ao.quantization.default_observer, weight=torch.ao.quantization.default_weight_observer
            )
    else:
        model.qconfig = qconfig
    print(f"Q config = {model.qconfig}")
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    torch.ao.quantization.prepare(model, inplace=True)
    
    from tqdm import tqdm
    print("calibrating...")
    for input,label in tqdm(iter(data),leave=False):
        _= model(input)
    torch.ao.quantization.convert(model, inplace=True)