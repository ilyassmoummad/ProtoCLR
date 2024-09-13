import torch
from tqdm import tqdm
from utils import model_dim
import torch.nn.functional as F

def val_knn(encoder, state_dict, train_loader, val_loader, len_trainset, val_transform, args):

    encoder.load_state_dict(state_dict)
    encoder.eval()

    train_features = torch.zeros(len_trainset, model_dim[args.model]).to(args.device)
    train_labels = torch.zeros(len_trainset).to(args.device)  

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            inputs, targets = batch["input_values"], batch["labels"]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                features = encoder(val_transform(inputs.to(args.device)))
            bs = inputs.size(0)
            features = F.normalize(features, dim=1)
            train_features[batch_idx * args.bs: batch_idx * args.bs + bs,:] = features.data
            train_labels[batch_idx * args.bs: batch_idx * args.bs + bs] = targets

    total = 0.
    correct = 0.

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            inputs, targets = batch["input_values"], batch["labels"]
            targets = targets.to(args.device)
            bs = inputs.size(0)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                features = encoder(val_transform(inputs.to(args.device)))
                dist = torch.mm(features, train_features.T)
            yd, yi = dist.topk(args.k, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(bs, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
        knn_acc = correct / total

    print('Val Acc: {}'.format(knn_acc))
        
    return knn_acc