from diaglib.learn.cnn.classify.models import AlexNet, VGG16, VGG19, ResNet50, ResNet152, BreakHisNet
from diaglib.learn.cnn.classify.models_v2 import VGG16v2, VGG19v2, ResNet50v2


MODEL_CLASSES = {
    'AlexNet': AlexNet,
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'ResNet152': ResNet152,
    'BreakHisNet': BreakHisNet,
    'VGG16v2': VGG16v2,
    'VGG19v2': VGG19v2,
    'ResNet50v2': ResNet50v2
}
