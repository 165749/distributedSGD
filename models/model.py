from models.alexnet import AlexNet
from models.mobilenet import MobileNetV2
from models.googlenet import GoogLeNet
from models.inception import Inception3
from models.resnet import Resnet50, Resnet101, Resnet152
from models.vgg import Vgg11, Vgg13, Vgg16, Vgg19

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
}
