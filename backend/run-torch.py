from channelexplorer import ChannelExplorer_Torch as Cexp
import torch
import torchvision.models as torch_models
import torchvision.datasets as torch_datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import transformers
from torchvision.transforms import functional as F
from torch.utils.data import Subset


MODEL, DATASET = ['vgg16'], ['image-dataset']
# MODEL, DATASET = ['vgg16', 'inceptionv3', 'GPT2'], ['image-dataset', 'image-dataset', 'GPT2-custom']
# MODEL, DATASET = ['vgg16', 'GPT2'], ['image-dataset', 'GPT2-custom']
# MODEL, DATASET = ['vgg16', 'inceptionv3'], ['image-dataset', 'image-dataset']
# MODEL, DATASET = ['vgg16'], ['image-dataset']
# MODEL, DATASET = ['vit', 'inceptionv3', 'vgg16', 'GPT2'], ['image-dataset', 'image-dataset', 'image-dataset', 'GPT2-custom']
# MODEL, DATASET = ['clip'], ['image-dataset']

models = []
for model_name in MODEL:
    if model_name == 'vgg16':
        model = torch_models.vgg16(weights=torch_models.VGG16_Weights.DEFAULT)
        model.eval()
        models.append(model)
    elif model_name == 'inceptionv3':
        model = torch_models.inception_v3(weights=torch_models.Inception_V3_Weights.DEFAULT)
        model.eval()
        models.append(model)
    elif model_name == 'GPT2':
        model = transformers.GPT2Model.from_pretrained('gpt2')
        model.config.pad_token_id = model.config.eos_token_id
        model.eval()
        models.append(model)
    elif model_name == 'vit':
        model = transformers.AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model.eval()
        models.append(model)
        
for model in models:
    print(model)
        
# Get the imagenette dataset
datasets = []
for dataset_name in DATASET:
    if dataset_name == 'image-dataset':
        dataset = torch_datasets.ImageFolder('/home/insane/U/apalysis-evaluation/dataset-5')
        datasets.append(dataset)
    elif dataset_name == 'GPT2-custom':
        dataset = [
            'Hello, how are you?',
            'I am fine, thank you!',
            'What is your name?',
            'My name is John Doe.',
            'How old are you?',
            'I am 25 years old.',
            'What is the capital of France?',
            'The capital of France is Paris.',
            'What is the capital of Germany?',
        ]
        datasets.append(dataset)

inputs = []
for model_name, model, dataset in zip(MODEL, models, datasets):
    if model_name == 'vgg16':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        sampled_images = Subset(dataset, np.random.choice(len(dataset), 10, replace=False).tolist())
        transformed_images = [transform(image) for image, _ in sampled_images]
        inputs.append(torch.stack(transformed_images))
    elif model_name == 'inceptionv3':
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        sampled_images = Subset(dataset, np.random.choice(len(dataset), 10, replace=False).tolist())
        transformed_images = [transform(image) for image, _ in sampled_images]
        inputs.append(torch.stack(transformed_images))
    elif model_name == 'GPT2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        inputs.append(tokenizer(dataset, return_tensors='pt', padding=True, truncation=True))
    elif model_name == 'vit':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        sampled_images = Subset(dataset, np.random.choice(len(dataset), 10, replace=False).tolist())
        transformed_images = [transform(image) for image, _ in sampled_images]
        inputs.append(torch.stack(transformed_images))
    else:
        raise ValueError(f"Model {model_name} not supported")


host = "localhost"
port = 8000
log_level = "info"

server = Cexp(
    models=models,
    all_inputs=inputs,
    log_level=log_level,
)

server.run_server(host=host, port=port)