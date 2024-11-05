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
from diffusers import StableDiffusionPipeline
from functools import lru_cache

import numpy as np
import random
from matplotlib import pyplot as plt

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Disable cuDNN benchmark to ensure deterministic behavior (slightly slower)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class ComplexNetWithBranch(torch.nn.Module):
    def __init__(self):
        super(ComplexNetWithBranch, self).__init__()
        
        # Common layers before branching
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU()
        )
        
        self.inner_seq_1 = torch.nn.Sequential(
            torch.nn.Linear(15, 15),
            torch.nn.ReLU()
        )
        self.inner_seq_2 = torch.nn.Sequential(
            torch.nn.Linear(15, 15),
            torch.nn.ReLU()
        )
        
        # Branch 1
        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(20, 15),
            self.inner_seq_1,
            self.inner_seq_2
        )
        
        # Branch 2
        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(20, 15),
            torch.nn.ReLU()
        )
        
        # Common layers after recombining
        self.fc_combine = torch.nn.Linear(30, 10)
        self.fc_output = torch.nn.Linear(10, 1)
        
        # ModuleList after recombination
        self.module_list = torch.nn.ModuleList([
            torch.nn.Linear(1, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ])
        
        self.branch_output = torch.nn.Linear(5, 5)
        self.branch_logit = torch.nn.ReLU()


    def forward(self, x):
        # Common forward pass before branching
        x = self.seq(x)
        
        # Branching
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        
        # Concatenate the outputs of both branches
        x = torch.cat([branch1_out, branch2_out], dim=1)
        
        # Pass through the combined layer
        x = self.fc_combine(x)
        x = self.fc_output(x)
        
        # ModuleList forward pass
        for i, layer in enumerate(self.module_list):
            x = layer(x)
            if i == 1:
                x2 = self.branch_output(x)
                x2 = self.branch_logit(x2)
        
        return x, x2



# MODEL, DATASET = ['ComplexNetWithBranch'], ['complex-dataset']
MODEL, DATASET = ['vgg16'], ['image-dataset']
# MODEL, DATASET = ['vgg16', 'inceptionv3'], ['image-dataset', 'image-dataset']
# MODEL, DATASET = ['inceptionv3'], ['image-dataset']
# MODEL, DATASET = ['vgg16', 'inceptionv3', 'GPT2'], ['image-dataset', 'image-dataset', 'GPT2-custom']
# MODEL, DATASET = ['vgg16', 'GPT2'], ['image-dataset', 'GPT2-custom']
# MODEL, DATASET = ['vgg16', 'inceptionv3'], ['image-dataset', 'image-dataset']
# MODEL, DATASET = ['vgg16'], ['image-dataset']
# MODEL, DATASET = ['vit', 'inceptionv3', 'vgg16', 'GPT2'], ['image-dataset', 'image-dataset', 'image-dataset', 'GPT2-custom']
# MODEL, DATASET = ['clip'], ['image-dataset']

# MODEL, DATASET = ['sd-text-encoder'], ['single-prompt']
# MODEL, DATASET = ['sd-text-encoder', 'sd-unet', 'sd-vae'], ['single-prompt', 'single-prompt', 'single-prompt']

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
    elif model_name == 'sd-text-encoder':
        model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').text_encoder
        model.eval()
        models.append(model)
    elif model_name == 'sd-unet':
        model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').unet
        model.eval()
        models.append(model)
    elif model_name == 'sd-vae':
        model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').vae
        model.eval()
        models.append(model)
    elif model_name == 'ComplexNetWithBranch':
        model = ComplexNetWithBranch()
        model.eval()
        models.append(model)
    else:
        raise ValueError(f"Model {model_name} not supported")
        
for model in models:
    print(model)
    
@lru_cache
def get_sd_pipeline():
    pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    pipeline.scheduler.set_timesteps(50)
    pipeline.scheduler.temperature = 0
    pipeline.safety_checker=lambda images, clip_input: (images, [False] * len(images))
    return pipeline
        
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
    elif dataset_name == 'single-prompt':
        dataset = 'A fantasy landscape with mountains and a river during sunset'
        datasets.append(dataset)
    elif dataset_name == 'complex-dataset':
        dataset = torch.randn(10, 10)
        datasets.append(dataset)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

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
    elif model_name == 'sd-text-encoder':
        pipeline = get_sd_pipeline()
        tokenizer = pipeline.tokenizer
        inputs.append(tokenizer(dataset, return_tensors='pt'))
    elif model_name == 'sd-unet':
        pipeline = get_sd_pipeline()
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder
        scheduler = pipeline.scheduler

        data = tokenizer(dataset, return_tensors='pt')
        with torch.no_grad():
            data = text_encoder(**data).last_hidden_state
        
        # Set the height and width of the latent image (1/8 of the final output size)
        height, width = 512, 512
        latents_shape = (1, model.in_channels, height // 8, width // 8)

        # Generate random noise
        latents = torch.randn(latents_shape, generator=torch.manual_seed(seed))

        # Scale the initial noise by the scheduler's initial standard deviation
        latents = latents * scheduler.init_noise_sigma
        
        inputs.append(data)
    elif model_name == 'sd-vae':
        inputs.append(dataset)
    elif model_name == 'ComplexNetWithBranch':
        inputs.append(dataset)
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



# input = ['a beautiful tree']

# pipeline = sd_model()
# tokenizer, text_encoder, scheduler, unet, vae = pipeline

# x = tokenizer(input)
# x = text_encoder(**x).last_hidden_state


# for step in scheduler.timesteps:
#     x = unet(x, step)
#     scheduler.step(x)
    
# output = vae.decode(x)
