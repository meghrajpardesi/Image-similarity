from PIL import Image
import torch
from torchvision import transforms


data_transforms = transforms.Compose([
    transforms.Resize(440),
    transforms.CenterCrop(416),
    transforms.ToTensor()
])


def image_loader(image_name :str) -> torch.Tensor:
    """image loader reads image from given path and return Tensor image."""
    image = Image.open(image_name)
    image = data_transforms(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

