import torch
import numpy as np
import pandas as pd
from utils.data_utils import DataloadingManager
from utils.eval_utils import plot_confusion_matrix, plot_curves
from train import ViTClassifier

def main():
    np.random.seed(0)
    torch.manual_seed(0)

    train_dataloader, test_dataloader, val_dataloader = DataloadingManager().get_dataloaders()

    model = ViTClassifier(get_test_config=True)
    # model.load_pretrained(r'out\best_model.pt')
    model.fit(train_dataloader,val_dataloader,log=False)

    test_metrics = model.evaluate(test_dataloader)
    val_metrics = model.evaluate(val_dataloader) 
    train_metrics = model.evaluate(train_dataloader)

    metrics = pd.DataFrame({'test':test_metrics, 'train':train_metrics, 'val':val_metrics})
    metrics.to_csv('out/metrics.csv')

    output, y_true = model.predict(test_dataloader)

    plot_confusion_matrix(output, y_true, 'out/cf_matrix.png' )
    plot_curves(output, y_true, 'out/curves.png')


if __name__ == '__main__':
    main()