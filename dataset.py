mport os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from preprocess import CustomPreprocess


transform = CustomPreprocess(size=(48, 48), normalize_mean=0.5, normalize_std=0.5)

def get_subset_loader(data_dir, transform, samples_per_class=5000, batch_size=32):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Select limited samples per class
    class_counts = defaultdict(int)
    selected_indices = []

    for idx, (_, label) in enumerate(dataset):
        
        if class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if all(count >= samples_per_class for count in class_counts.values()):
            break

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader, dataset.class_to_idx

# Paths to your folders
train_path = 'Data/train'
test_path = 'Data/test'

# Loaders
train_loader, train_class_map = get_subset_loader(train_path, transform)
test_loader, test_class_map = get_subset_loader(test_path, transform)

print("Train classes:", train_class_map)
print("Test classes:", test_class_map)

print(len(train_loader))  
