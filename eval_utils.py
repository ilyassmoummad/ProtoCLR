from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import torch

def sample_episode(features, labels, K):

    unique_classes = torch.unique(labels)
    
    support_set = []
    query_set = []
    support_labels = []
    query_labels = []
    
    for cls in unique_classes:
        # Get indices of all examples of the current class
        cls_indices = torch.where(labels == cls)[0]

        # Shuffle the indices
        selected_indices = cls_indices[torch.randperm(len(cls_indices))[:]]
        
        # Split into support (K examples) and query sets (Max-K examples)
        support_indices = selected_indices[:K]
        query_indices = selected_indices[K:]
        
        # Collect support set
        support_set.append(features[support_indices])
        support_labels.append(labels[support_indices])
        
        # Collect query set
        query_set.append(features[query_indices])
        query_labels.append(labels[query_indices])
    
    # Stack the list of tensors into a single tensor
    support_set = torch.cat(support_set, dim=0)
    query_set = torch.cat(query_set, dim=0)
    support_labels = torch.cat(support_labels, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    
    return support_set, support_labels, query_set, query_labels

def nearest_prototype(support_set, support_labels, query_set, query_labels):

    support_set = F.normalize(support_set, dim=1)
    query_set = F.normalize(query_set, dim=1)

    support_mean = support_set.mean(dim=0)
    support_set -= support_mean
    query_set -= support_mean

    unique_labels = torch.unique(support_labels)

    support_prototypes = []
    for label in unique_labels:
        support_prototypes.append(support_set[support_labels == label].mean(dim=0))
    support_prototypes = torch.stack(support_prototypes)
    distances = torch.cdist(query_set, support_prototypes)
    query_pred_indices = distances.argmin(dim=1)
    query_predictions = unique_labels[query_pred_indices]

    accuracy = (query_predictions == query_labels).float().mean().detach()

    return accuracy

def fewshot_test(encoder, loader, transform, args, shot=1):
    
    encoder.eval()

    features = []
    labels = []

    for idx, batch in enumerate(tqdm(loader)):

        x, y = batch

        x = x.to(args.device)
        y = y.to(args.device)

        with torch.no_grad():
            
            x = transform(x)
            z = encoder(x)
            
        features.append(z)
        labels.append(y)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    

    accs = []

    for i in range(args.nruns):
        support_set, support_labels, query_set, query_labels = sample_episode(features, labels, shot)
        accuracy = nearest_prototype(support_set, support_labels, query_set, query_labels)
        accs.append(accuracy)

    accs = torch.tensor(accs)
    acc = accs.mean()
    std = accs.std()

    return acc, std

def fewshot_val(encoder, dataset, transform, args, shot=1):
    
    encoder.eval()

    audios = dataset['data'].to(args.device)
    targets = dataset['label'].to(args.device)

    tensor_dataset = TensorDataset(audios, targets)

    loader = DataLoader(tensor_dataset, batch_size=args.bs, shuffle=True, drop_last=False)  

    accs = []

    features = []
    labels = []

    for idx, batch in enumerate(tqdm(loader)):

        x, y = batch

        x = x.to(args.device)
        y = y.to(args.device)

        with torch.no_grad():
            
            x = transform(x)
            z = encoder(x)
            
        features.append(z)
        labels.append(y)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    for i in range(args.nruns):
        support_set, support_labels, query_set, query_labels = sample_episode(features, labels, shot)
        accuracy = nearest_prototype(support_set, support_labels, query_set, query_labels)
        accs.append(accuracy)

    accs = torch.tensor(accs)
    acc = accs.mean()
    std = accs.std()

    return acc, std