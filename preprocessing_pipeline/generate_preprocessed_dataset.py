import os
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from PIL import Image
from preprocess_pipeline import RemoveSpecularHighlights, LightingNormalization


class PreprocessPipeline:
    def __init__(self,
                 batch_size: int = 128,
                 num_workers: int = 2,
                 transform: str = 'spec'):
        
        """
        Pipeline para o pré-processamento do dataset de imagens de colonoscopia. Gera um novo dataset

        Parâmetros:
            batch_size e num_workers: parâmetros da classe Dataloader do PyTorch
            transform: determina quais etapas de pré-processamento serão aplicadas às imagens. Pode ser:
                spec: aplica apenas a remoção de picos de luz (specular highlights) - recomendado
                both: aplica remoção de picos de luz e normalização de iluminação da imagem
        """
        
        assert transform in ['spec','both'], 'transform should be either \'spec\' or \'both\''

        self.path_a = r'dataset\CP-CHILD-A'
        self.path_b = r'dataset\CP-CHILD-B'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.toTensor = transforms.ToTensor()


    def load_images(self):
        train_path_a = self.path_a + r'\Train'
        test_path_a = self.path_a + r'\Test'
        train_path_b = self.path_b + r'\Train'
        test_path_b = self.path_b + r'\Test'

        train_dataset_a = datasets.ImageFolder(train_path_a,transform=self.toTensor)
        test_dataset_a = datasets.ImageFolder(test_path_a,transform=self.toTensor)
        train_dataset_b = datasets.ImageFolder(train_path_b,transform=self.toTensor)
        test_dataset_b = datasets.ImageFolder(test_path_b,transform=self.toTensor)

       
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_a,train_dataset_b]) # type:ignore
        test_dataset = torch.utils.data.ConcatDataset([test_dataset_a,test_dataset_b]) # type:ignore

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, 
                                        shuffle=True, num_workers=self.num_workers)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, 
                                        shuffle=True, num_workers=self.num_workers)
        
        return train_dataloader, test_dataloader



    def save_images(self,
                    image: np.ndarray,
                    label: torch.Tensor,
                    dataset: str,
                    images_processed: int):
        
        assert dataset in ['Train','Test'], 'dataset should be either  \'Train\' or \'Test\''

        polyp_path = 'preprocessed_dataset/' + dataset + '/Polyp/'
        non_polyp_path = 'preprocessed_dataset/' + dataset + '/Non-Polyp/'


        if not os.path.exists('preprocessed_dataset/' + dataset):
            os.makedirs(polyp_path)
            os.makedirs(non_polyp_path)

        im = Image.fromarray(image)

        save_path = polyp_path + f'{images_processed}.jpeg' if bool(label) else non_polyp_path + f'{images_processed}.jpeg'
        im.save(save_path)


    def generate_preprocessed_dataset(self):
        
        train_dataloader, test_dataloader = self.load_images()
        images_processed = 0

        print('========= Processando dataset de treino ========\n\n')

        for features, labels in train_dataloader:
            for i, image in enumerate(features):
                img = RemoveSpecularHighlights()(image)
                self.save_images(LightingNormalization(patch_size=4)(img) if self.transform == 'both' else img,
                                labels[i],
                                dataset='Train',
                                images_processed=images_processed)
                images_processed += 1
                if images_processed % 100 == 0:
                    print(f'Imagens processadas: {images_processed}\n')

        print('\nDataset de treino completo\n\n')

        images_processed = 0

        print('========= Processando dataset de teste ========\n\n')

        for features, labels in test_dataloader:
            for i, image in enumerate(features):
                img = RemoveSpecularHighlights()(image)
                self.save_images(LightingNormalization()(img) if self.transform == 'both' else img,
                                labels[i],
                                dataset='Test',
                                images_processed=images_processed)                
                images_processed += 1
                if images_processed % 100 == 0:
                    print(f'Imagens processadas: {images_processed}\n')

        print('\nDataset de teste completo\n\n')

if __name__ == '__main__':
    preprocess = PreprocessPipeline(transform='spec')
    preprocess.generate_preprocessed_dataset()