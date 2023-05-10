import os
import torch
import numpy as np
import pandas as pd
from utils.data_utils import DataloadingManager
from utils.eval_utils import plot_confusion_matrix, plot_curves
from train import ViTClassifier

def main(dataset, out_dir):
    np.random.seed(0)
    torch.manual_seed(0)

    # out_dir = 'out/specnorm'
    # dataset = 'cp-child-specnorm'

    train_dataloader, test_dataloader, val_dataloader = DataloadingManager(dataset_name=dataset).get_dataloaders()

    model = ViTClassifier(get_test_model=False, get_test_config=False)
    # model.load_pretrained('out/no_preprocessing/best_model.pt')
    model.fit(train_dataloader, val_dataloader, log=False, resume_training=False, save_path=os.path.join(out_dir, 'best_model.pt'))

    test_metrics = model.evaluate(test_dataloader)
    val_metrics = model.evaluate(val_dataloader) 
    train_metrics = model.evaluate(train_dataloader)

    metrics = pd.DataFrame({'test':test_metrics, 'train':train_metrics, 'val':val_metrics})
    metrics.to_csv(os.path.join(out_dir, 'metrics.csv'))

    output, y_true = model.predict(test_dataloader)

    plot_confusion_matrix(output, y_true, os.path.join(out_dir, 'cf_matrix.png'))
    plot_curves(output, y_true, os.path.join(out_dir, 'curves.png'))


if __name__ == '__main__':
    main('cp-child', 'out/no_preprocessing')
    # main('cp-child-specnorm', 'out/specnorm')