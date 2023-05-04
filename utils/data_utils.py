import os
import torch

from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.configs import get_loading_config


def get_mean_and_std(dataloader):
    '''
    Get mean and standard deviation for normalization
    '''

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std



class DataloadingManager:
    def __init__(self, 
                 dataset_name: str = 'cp-child-specnorm', 
                 mean: list[float] = [0.5, 0.5, 0.5],
                 std: list[float] = [0.5, 0.5, 0.5]):

        options = ['cp-child','cp-child-subsampled','cp-child-specnorm','cp-child-specnorm-lightnorm']
        assert dataset_name in options, f'Dataset should be one of: {options}'
        
        self.dataset_name = dataset_name
        self.mean = mean
        self.std = std
        self.train_configs = get_loading_config()

        if dataset_name == 'cp-child':
            '''
            Standard dataset
            '''
            self.path = 'datasets/original_dataset'

        elif dataset_name == 'cp-child-specnorm':
            '''
            Dataset with specular highlights removed
            '''
            self.path = 'datasets/preprocessed_dataset_specnorm'

        elif dataset_name == 'cp-child-specnorm-lightnorm':
            '''
            Dataset with both specular highlights removal and lightning normalization preprocessing steps applied 
            '''
            self.path = 'datasets/preprocessed_dataset_specnorm'

        elif dataset_name == 'cp-child-subsampled':
            '''
            Dataset with an equal number of polyp and non-polyp images
            '''
            self.path = 'datasets/subsampled_dataset'

    

    def __repr__(self):
        train_polyp_len = len([name for name in os.listdir(os.path.join(self.path,'Train','Polyp'))])
        train_npolyp_len = len([name for name in os.listdir(os.path.join(self.path,'Train','Non-Polyp'))])
        test_polyp_len = len([name for name in os.listdir(os.path.join(self.path,'Train','Polyp'))])
        test_npolyp_len = len([name for name in os.listdir(os.path.join(self.path,'Test','Non-Polyp'))])

        rep_string = f'''
        Dataset {self.dataset_name}: 
            {train_polyp_len+train_npolyp_len} training images ({train_polyp_len} polyp, {train_npolyp_len} non-polyp)
            {test_polyp_len+test_npolyp_len} test images ({test_polyp_len} polyp, {test_npolyp_len} non-polyp)
        '''
        return rep_string



    def __load_standard_dataset(self, batch_size, val_ratio, num_workers):
        '''
        Util for the 'cp-child' dataset. Concatenates and loads datasets CP-CHILD-A and CP-CHILD-B
        Only used if self.dataset_name == 'cp-child'
        '''
        path_a = 'datasets/original_dataset/CP-CHILD-A'
        path_b = 'datasets/original_dataset/CP-CHILD-B'

        train_path_a = os.path.join(path_a,'Train')
        test_path_a = os.path.join(path_a,'Test')
        train_path_b = os.path.join(path_b,'Train')
        test_path_b = os.path.join(path_b,'Test')

        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)
                                        ])

        train_dataset_a = datasets.ImageFolder(train_path_a,transform=transform)
        test_dataset_a = datasets.ImageFolder(test_path_a,transform=transform)
        train_dataset_b = datasets.ImageFolder(train_path_b,transform=transform)
        test_dataset_b = datasets.ImageFolder(test_path_b,transform=transform)

        train_dataset = ConcatDataset([train_dataset_a,train_dataset_b])
        test_dataset = ConcatDataset([test_dataset_a,test_dataset_b])

        test_dataset, val_dataset = random_split(test_dataset, [int((1-val_ratio)*len(test_dataset)),
                                                                int(val_ratio*len(test_dataset))])

        print(f"Size of training set: {len(train_dataset)}")
        print(f"Size of test set: {len(test_dataset)}")
        print(f"Size of validation set: {len(val_dataset)}\n")

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers)

        return train_dataloader, test_dataloader, val_dataloader



    def get_dataloaders(self):
        '''
        Loads dataset with sampler
        '''
        VAL_RATIO = self.train_configs['val_ratio']
        BATCH_SIZE = self.train_configs['batch_size']
        NUM_WORKERS = self.train_configs['num_workers']

        if self.dataset_name == 'cp-child':
            return self.__load_standard_dataset(BATCH_SIZE, VAL_RATIO, NUM_WORKERS)
        
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)
                                        ])
        
        train_path = os.path.join(self.path, 'Train')
        test_path = os.path.join(self.path, 'Test')

        train_dataset = datasets.ImageFolder(train_path,transform=transform)
        test_dataset = datasets.ImageFolder(test_path,transform=transform)

        test_dataset, val_dataset = random_split(test_dataset, [int((1-VAL_RATIO)*len(test_dataset)),
                                                                    int(VAL_RATIO*len(test_dataset))])
        
        # Get class counts for sampling
        ds_size = len(train_dataset)
        counts = torch.unique(torch.Tensor(train_dataset.targets), return_counts=True)[1]
        sample_weights = [1-(counts[i]/ds_size) for i in train_dataset.targets]

        sampler = WeightedRandomSampler(weights = sample_weights, 
                                        num_samples = ds_size,
                                        replacement = True)

        print(f"Size of training set: {len(train_dataset)}")
        print(f"Size of test set: {len(test_dataset)}")
        print(f"Size of validation set: {len(val_dataset)}\n")

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                                        sampler=sampler, num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, 
                                        shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, 
                                        shuffle=True, num_workers=NUM_WORKERS)

        return train_dataloader, test_dataloader, val_dataloader