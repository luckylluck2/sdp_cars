import os
from torch.nn import CrossEntropyLoss
import numpy as np
import torch.optim as optim
from time import sleep
from tqdm import tqdm
from torch import save
from torch.nn.functional import softmax


def calculate_metrics(tp, tn, fp, fn):
    """Calculates a number of metrics based on the number of true positives,
    true negatives, false positives and false negatives.
    Args:
        tp (int): Number of true positives
        tn (int): Number of true negatives
        fp (int): Number of false positives
        fn (int): Number of false negatives
    Returns:
        precision (float): Fraction of positive classifications that were correct
        recall (float): Fraction of positive examples that were correctly classified (true positive rate)
        selectivity (float): Fraction of negative examples that were correctly classified (true negative rate)
        accuracy (float): Fraction of examples that were correctly classified
        balanced_accuracy (float): Mean of the recall and selectivity
        f1 (float): F1 score, harmonic mean of the precision and recall
    """
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    selectivity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (recall + selectivity) / 2
    f1 = 2 * tp / (2 * tp + fp + fn)

    metric_dict = {'precision': precision,
                   'recall': recall,
                   'selectivity': selectivity,
                   'accuracy': accuracy,
                   'balanced_accuracy': balanced_accuracy,
                   'f1': f1}

    return metric_dict


def step(data_set, model, optimizer, criterion, epoch, mode='train'):
    losses = []
    data_size = len(data_set)
    progress_bar = tqdm(np.arange(data_size), desc=f'{mode} epoch {epoch}', total=data_size)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    if mode == 'train':
        model.train()
    else:
        model.eval()

    for i, (x, y) in enumerate(data_set):
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item())
        
        # predictions = softmax(out, 1).detach().numpy()[:, 1] > 0.5
        # truths = y.detach().numpy()[:, 1] > 0.5
        # output_pairs = np.stack((predictions, truths), axis=-1)
        
        # true_positives += np.sum(np.all(output_pairs == np.array([True, True]), axis=1))
        # true_negatives += np.sum(np.all(output_pairs == np.array([False, False]), axis=1))
        # false_positives += np.sum(np.all(output_pairs == np.array([True, False]), axis=1))
        # false_negatives += np.sum(np.all(output_pairs == np.array([False, True]), axis=1))
        # predictions = softmax(out, 1).detach().numpy()[:, 1] > 0.5
        # truths = y.detach().numpy()[:, 1] > 0.5
        # output_pairs = np.stack((predictions, truths), axis=-1)

        # true_positives += np.sum(np.all(output_pairs == np.array([True, True]), axis=1))
        # true_negatives += np.sum(np.all(output_pairs == np.array([False, False]), axis=1))
        # false_positives += np.sum(np.all(output_pairs == np.array([True, False]), axis=1))
        # false_negatives += np.sum(np.all(output_pairs == np.array([False, True]), axis=1))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del x, y, loss
        else:
            del x, y, loss
        if i < len(data_set) - 1:
            progress_bar.update()

    loss = np.mean(losses)
    # metrics = calculate_metrics(true_positives, true_negatives, false_positives, false_negatives)

    metrics = {'precision': 0,
                   'recall': 0,
                   'selectivity': 0,
                   'accuracy': 0,
                   'balanced_accuracy': 0,
                   'f1': 0}
    
    progress_bar.set_postfix({'loss': loss, 
                              'tp': true_positives,
                              'tn': true_negatives,
                              'fp': false_positives,
                              'fn': false_negatives,})
    progress_bar.update()

    return loss, metrics


def train_model(model, model_save_dir, train_dataset, val_dataset=None, num_epochs=40, lr=1e-4, 
                early_stopping_threshold=10, l2=1e-4):
    model_parameters_path = os.path.join(model_save_dir, 'model_parameters')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    criterion = CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam([{'params': model.parameters()}], lr=lr, weight_decay=l2)

    min_loss = np.inf
    epochs_without_improvement = 0
    
    metrics_dict = {'train_loss': [], 'val_loss': [],
                    'train_precision': [], 'val_precision': [],
                    'train_recall': [], 'val_recall': [],
                    'train_selectivity': [], 'val_selectivity': [],
                    'train_accuracy': [], 'val_accuracy': [],
                    'train_balanced_accuracy': [], 'val_balanced_accuracy': [],
                    'train_f1': [], 'val_f1': []}

    for epoch in range(num_epochs):
        for mode, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
            if dataset is not None:
                loss, metrics = step(dataset, model, optimizer, criterion, epoch, mode)

                metrics_dict[f'{mode}_loss'].append(loss)
                metrics_dict[f'{mode}_precision'].append(metrics['precision'])
                metrics_dict[f'{mode}_recall'].append(metrics['recall'])
                metrics_dict[f'{mode}_selectivity'].append(metrics['selectivity'])
                metrics_dict[f'{mode}_accuracy'].append(metrics['accuracy'])
                metrics_dict[f'{mode}_balanced_accuracy'].append(metrics['balanced_accuracy'])
                metrics_dict[f'{mode}_f1'].append(metrics['f1'])
        
        # Prevent printing of progress bars from messing up
        print(' ')
        sleep(0.1)

        # Save model state if it has improved
        if loss < min_loss:
            save(model.state_dict(), model_parameters_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        min_loss = min(min_loss, loss)

        # Stop training early if the model has not improved for 'early_stopping_threshold' epochs.
        if epochs_without_improvement >= early_stopping_threshold:
            print(f'Early stopping after {epoch} epochs')
            break
    
    return metrics_dict
