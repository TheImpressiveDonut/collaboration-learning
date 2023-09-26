from efficientnet_pytorch import EfficientNet

from models.netcnn import NetCNN
from models.netfnn import NetFNN
from models.resnet import ResNet, resnet20


def select_client_model(dataset_name: str, model_name: str) -> [ResNet | NetCNN | EfficientNet]:
    if dataset_name[0:7] == 'cifar10':
        if model_name == 'resnet':
            return resnet20(num_classes=args.num_classes)
        elif model_name == 'cnn':
            return NetCNN(in_features=args.num_channels, num_classes=args.num_classes, dim=1600)
        elif model_name == 'fnn':
            return NetFNN(input_dim=3 * 32 * 32, mid_dim=100, num_classes=args.num_classes)
        else:
            raise NotImplementedError
    elif dataset_name == 'MNIST':
        return NetCNN(in_features=1, num_classes=10, dim=1024)
    elif dataset_name in ['fed-isic-2019', 'fed-isic-2019-new']:
        if model_name == 'efficient-net':
            return EfficientNet.from_pretrained(args.arch_name, num_classes=args.num_classes)
        elif model_name == 'cnn':
            return NetCNN(in_features=args.num_channels, num_classes=args.num_classes, dim=10816)
        elif model_name == 'fnn':
            return NetFNN(input_dim=3 * 64 * 64, mid_dim=100, num_classes=8)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
