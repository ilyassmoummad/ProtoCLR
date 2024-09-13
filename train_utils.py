import torch.nn.functional as F
from val_utils import val_knn
from tqdm import tqdm
import torch
import math

def cosine_lr_scheduler(optimizer, epoch, args):
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_epoch(encoder, projector, train_loader, train_transform, loss_fn, optim, scaler, epoch, args):

    encoder.train()
    projector.train()

    print("Epoch {}".format(epoch+1))
    cosine_lr_scheduler(optim, epoch, args)

    tr_loss = 0.

    if args.loss == 'ce':

        for idx, batch in enumerate(tqdm(train_loader)):

            x, y = batch["input_values"], batch["labels"]

            optim.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                x = x.to(args.device)
                y = y.to(args.device)

                with torch.no_grad():
                    
                    x = train_transform(x)

                z = encoder(x)

                h = projector(z)

                loss = loss_fn(h, y)

            tr_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

    else:

        for idx, batch in enumerate(tqdm(train_loader)):

            x, y = batch["input_values"], batch["labels"]

            optim.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                x = x.to(args.device)
                y = y.to(args.device)

                with torch.no_grad():

                    x1 = train_transform(x); x2 = train_transform(x)

                z1 = encoder(x1); z2 = encoder(x2)

                h1 = projector(z1); h2 = projector(z2)

                if args.loss in ['supcon', 'protoclr']:
                    loss = loss_fn(h1, h2, y)
                
                elif args.loss  == 'simclr':
                    loss = loss_fn(h1, h2) 
        
            tr_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

    tr_loss = tr_loss/len(train_loader)
    print('Train Loss: {}'.format(tr_loss))

    encoder_state_dict = encoder.state_dict()
    if args.loss == 'ce':
        projector_state_dict = projector.state_dict()
        return encoder_state_dict, projector_state_dict, tr_loss
    else:
        return encoder_state_dict, tr_loss

def train(encoder, projector, train_loader, knn_train_loader, knn_val_loader, len_trainset, train_transform, val_transform, loss_fn, optim, args):

    print(f"Training starting on {args.device}")

    num_epochs = args.epochs

    encoder = encoder.to(args.device)
    projector = projector.to(args.device)
    
    scaler = torch.cuda.amp.GradScaler()

    if args.loss == 'ce':
        best_acc = 0.
    
    for epoch in range(num_epochs):
        if args.loss == 'ce':
            encoder_state_dict, projector_state_dict, tr_loss = train_epoch(encoder, projector, train_loader, train_transform, loss_fn, optim, scaler, epoch, args)
        else:
            encoder_state_dict, tr_loss = train_epoch(encoder, projector, train_loader, train_transform, loss_fn, optim, scaler, epoch, args)
        val_acc = val_knn(encoder, encoder_state_dict, knn_train_loader, knn_val_loader, len_trainset, val_transform, args)

        if args.loss == 'ce':
            if val_acc > best_acc:
                best_acc = val_acc
                best_state_dict_encoder = encoder_state_dict
                best_state_dict_projector = projector_state_dict

    last_state_dict = encoder_state_dict

    if args.loss == 'ce':

        return best_state_dict_encoder, best_state_dict_projector
    else:
        return last_state_dict