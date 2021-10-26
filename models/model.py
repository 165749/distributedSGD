from models.alexnet import AlexNet
from models.mobilenet import MobileNetV2
from models.googlenet import GoogLeNet
from models.inception import Inception3
from models.resnet import Resnet50, Resnet101, Resnet152
from models.vgg import Vgg11, Vgg13, Vgg16, Vgg19
from models.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, \
    EfficientNetB5, EfficientNetB6, EfficientNetB7
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201


name_to_model = {
    "alexnet": AlexNet,
    "mobilenet": MobileNetV2,
    "googlenet": GoogLeNet,
    "inception3": Inception3,
    "resnet50": Resnet50,
    "resnet101": Resnet101,
    "resnet152": Resnet152,
    "vgg11": Vgg11,
    "vgg13": Vgg13,
    "vgg16": Vgg16,
    "vgg19": Vgg19,
    "efficientnet-b0": EfficientNetB0,
    "efficientnet-b1": EfficientNetB1,
    "efficientnet-b2": EfficientNetB2,
    "efficientnet-b3": EfficientNetB3,
    "efficientnet-b4": EfficientNetB4,
    "efficientnet-b5": EfficientNetB5,
    "efficientnet-b6": EfficientNetB6,
    "efficientnet-b7": EfficientNetB7,
    "densenet121": DenseNet121,
    "densenet161": DenseNet161,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
}
