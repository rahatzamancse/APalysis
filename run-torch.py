from activation_pathway_analysis_backend.apalysis_torch import APAnalysisTorchModel
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from nltk.corpus import wordnet as wn

# MODEL, DATASET = 'inceptionv3', 'imagenet'
# MODEL, DATASET = 'vgg16', 'imagenet'

# MODEL, DATASET = 'inceptionv3', 'imagenette'
MODEL, DATASET = 'vgg16', 'mnist'

# MODEL, DATASET = 'simple_cnn', 'mnist'

# MODEL, DATASET = 'expression', 'fer2023'

# Load a demo model
if MODEL == 'vgg16':
    model = models.vgg16(pretrained=True)
elif MODEL == 'inceptionv3':
    model = models.inception_v3(pretrained=True)
elif MODEL == 'simple_cnn':
    # TODO
    model = None
elif MODEL == 'expression':
    # TODO
    model = None
else:
    raise ValueError(f"Model {MODEL} not supported")

# Define loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# Load dataset
if DATASET == 'imagenet':
    # TODO: Download Imagenet and point to path. write Transform function
    # https://image-net.org/download-images
    dataset = datasets.ImageNet(
        "path/to/downloaded/imagenet",
        split='train'
    )
    labels = None
elif DATASET == 'imagenette':
    # TODO
    dataset = None
    labels = None
elif DATASET == 'mnist':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    dataset = datasets.MNIST('/run/media/insane/SSD Games/Pytorch', train=True, download=True, transform=transform)
    labels = [str(i) for i in range(10)]
elif DATASET == 'fer2023':
    # TODO
    dataset = None
    labels = None

host = "localhost"
port = 8000
log_level = "info"

server = APAnalysisTorchModel(
    model=model,
    input_shape=(1, 3, 224, 224),
    dataset=dataset,
    label_names=labels,
    log_level=log_level,
)

server.run_server(host=host, port=port)