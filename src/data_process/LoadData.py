from src.data_process.ThyroidCancerDataset import ThyroidCancerDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def load_data(data_dir:str, batch_size:int=32, num_workers:int=4, classes:dict={0: ["B2"], 1: ["B5"], 2: ["B6"]}, balance_train_set=True, num_x=100):
    """
    Function to load and preprocess data for a machine learning model.

    Parameters:
    data_dir (str): The directory where the data is stored.
    batch_size (int): The number of samples per batch.
    num_workers (int): The number of subprocesses to use for data loading.
    classes (dict): A dictionary mapping class indices to class labels.

    Returns:
    tuple: A tuple containing the training, validation, and test data loaders.
    """
    print('Creating dataset...')
    train_dataset = ThyroidCancerDataset(data_dir=data_dir, transform=None, classes=classes, balance=balance_train_set, mode='train', num_x=num_x)
    print('Train dataset size:', len(train_dataset))

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    valid_dataset = ThyroidCancerDataset(data_dir=data_dir, transform=transform, classes=classes, balance=False, mode='valid')
    print('Valid dataset size:', len(valid_dataset))

    test_dataset = ThyroidCancerDataset(data_dir=data_dir, transform=transform, classes=classes, balance=False, mode='test')
    print('Test dataset size:', len(test_dataset))

    print('Creating dataloader...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('Train loader size:', len(train_loader))

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('Valid loader size:', len(valid_loader))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('Test loader size:', len(test_loader))
    
    return train_loader, valid_loader, test_loader