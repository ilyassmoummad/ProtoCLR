from eval_utils import fewshot_val
from tqdm import tqdm
import torch
import math
import os

def cosine_lr_scheduler(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # cosine lr schedule
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

            x, y = batch

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

            x, y = batch

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
                
                elif args.loss == 'simclr':
                    loss = loss_fn(h1, h2) 
        
            tr_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

    tr_loss = tr_loss/len(train_loader)
    print('Train Loss: {}'.format(tr_loss))

    if args.loss == 'ce':
        return encoder, projector, tr_loss
    else:
        return encoder, tr_loss

def train(encoder, projector, train_loader, val_dataset, train_transform, val_transform, loss_fn, optim, args):

    print(f"Training starting on {args.device}")

    num_epochs = args.epochs

    encoder = encoder.to(args.device)
    projector = projector.to(args.device)
    
    scaler = torch.cuda.amp.GradScaler()

    if args.loss == 'ce':
        best_acc = 0.
    
    for epoch in range(num_epochs):
        if args.loss == 'ce':
            encoder, projector, tr_loss = train_epoch(encoder, projector, train_loader, train_transform, loss_fn, optim, scaler, epoch, args)
        else:
            encoder, tr_loss = train_epoch(encoder, projector, train_loader, train_transform, loss_fn, optim, scaler, epoch, args)
        val_acc, val_std = fewshot_val(encoder, val_dataset, val_transform, args, shot=1)
        print(f"Val Acc: {val_acc}+-{val_std}")

        if args.loss == 'ce':
            if val_acc > best_acc:
                best_acc = val_acc
                best_state_dict_encoder = encoder.state_dict()
                best_state_dict_projector = projector.state_dict()

        if args.savefreq:   
            if (epoch+1) % args.freq == 0:
                os.makedirs(args.modelpath, exist_ok=True) 
                best_model_path = os.path.join(args.modelpath, args.model + '_' + args.loss + '_pretrain_' + args.pretrainds + '_epochs' + str(args.epochs) + '_lr' + str(args.lr) + '_trial1.pth')
                if os.path.isfile(best_model_path):
                    i = 1
                    while os.path.isfile(best_model_path):
                        i += 1
                        best_model_path = best_model_path[:-5] + str(i) + '.pth'
                strg = 'epochs' + str(args.epochs)
                newstrg = 'epoch' + str(epoch+1)
                freq_model_path = best_model_path.replace(strg, newstrg)

                if args.loss == 'ce':
                    torch.save({'encoder': best_state_dict_encoder, 'classifier': best_state_dict_projector}, freq_model_path)
                else:
                    torch.save(encoder.state_dict(), freq_model_path)

    if args.loss == 'ce':
        return best_state_dict_encoder, best_state_dict_projector
    else:
        return encoder.state_dict()