import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, dataloader

cuda_available = torch.cuda.is_available()

# standard training image configurations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize( # standard imagenet normalisation
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

data_directory = "./rubber_leaf_dataset_mendeley/raw_dataset"
dataset = datasets.ImageFolder(data_directory, transform=train_transforms)

# create the dataloader
data_loader = dataloader.DataLoader(
    dataset=dataset,
    batch_size=32, # standard config
    shuffle=True,
    num_workers=4,
    pin_memory=cuda_available
)

if __name__ == "__main__":
    print(class_to_idx := dataset.class_to_idx)

    for images, labels in data_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break