import torch
import numpy as np
import time
from collections import OrderedDict
from collections import defaultdict

def copy_state(model):
    """
    Given a PyTorch module `model`, makes a copy of the state onto the CPU.

    Parameters:
    ----------
    model : torch.nn.Module
        PyTorch module from which to copy the state dictionary.

    Returns:
    -------
    copy_dict : collections.OrderedDict
        A copy of the state dictionary with all tensors allocated on the CPU.
    """
    copy_dict = OrderedDict()
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        copy_dict[k] = v.cpu() if v.is_cuda else v.clone()
    return copy_dict


def test_epoch(model, img_test, batch_size=100):
    model.eval()
    n_test = img_test.shape[0]
    spks_test_pred = []
    with torch.no_grad():
        for k in np.arange(0, n_test, batch_size):
            kend = min(k+batch_size, n_test)
            img_batch = img_test[k:kend]
            spks_pred = model(img_batch)
            spks_test_pred.append(spks_pred.detach().cpu().numpy())
            # spks_test_pred[k:kend] = spks_pred
    test_pred = np.vstack(spks_test_pred)
    return test_pred

def val_epoch(model, val_dl, dl_length, n_neurons, batch_size, l1_readout=0, l2_readout=0, device=torch.device('cuda'), parallel=False, hs_reg=0.0):
    model.eval()

    test_loss = 0
    spks_test_pred = torch.zeros((dl_length, n_neurons), device=device)
    spks_test = torch.zeros((dl_length, n_neurons), device=device)
    # spks_test_gpu = spks_test.to(device)
    index_array = np.arange(0, dl_length, batch_size)
    with torch.no_grad():
        for k , (_, batch) in zip(index_array, val_dl):
            spks_batch = batch["responses"]
            img_batch = batch["screen"]
            spks_batch = spks_batch.to(device).squeeze()        # Right now we dont use chunk_size in cfg_val.dataset.modality_config.screen.chunk_size
            img_batch = img_batch.to(device).squeeze()
            img_batch = img_batch.unsqueeze(1)                             # add a channel dim

            spks_pred = model(img_batch)

            # compute loss
            if parallel:
                loss = model.module.loss_function(spks_batch, spks_pred, l1_readout=l1_readout, l2_readout=l2_readout, hs_reg=hs_reg)
            else:
                loss = model.loss_function(spks_batch, spks_pred, l1_readout=l1_readout, l2_readout=l2_readout, hs_reg=hs_reg)
            test_loss += loss.item()
            
            kend = min(k+batch_size, dl_length)
            spks_test[k:kend] = spks_batch
            spks_test_pred[k:kend] = spks_pred
        test_loss /=  dl_length

    varexp = ((spks_test - spks_test_pred)**2).sum(axis=0) 
    varexp /= ((spks_test-spks_test.mean(axis=0))**2).sum(axis=0)
    varexp = 1 - varexp
    test_pred = spks_test_pred.detach().clone()

    return test_loss, varexp, test_pred

def train_epoch(model, optimizer, train_dl, dl_length, epoch=0, batch_size=100, l1_readout=0, \
    device = torch.device('cuda'), detach_core=False, clamp=True, parallel=False, hs_reg=0.0):
    np.random.seed(epoch)

    train_loss = 0
    for _, batch in train_dl:
        optimizer.zero_grad()

        spks_batch = batch["responses"]
        img_batch = batch["screen"]
        spks_batch = spks_batch.to(device).squeeze()        # Right now we dont use chunk_size in cfg_val.dataset.modality_config.screen.chunk_size
        img_batch = img_batch.to(device).squeeze()
        img_batch = img_batch.unsqueeze(1)                             # add a channel dim

        spks_pred = model(img_batch, detach_core=detach_core)
        if parallel:
            loss = model.module.loss_function(spks_batch, spks_pred, l1_readout=l1_readout, hs_reg=hs_reg)
        else:
            loss = model.loss_function(spks_batch, spks_pred, l1_readout=l1_readout, hs_reg=hs_reg)
        loss.backward()
        optimizer.step()
        if clamp:
            if parallel:
                model.module.readout.Wx.data.clamp_(0)
                model.module.readout.Wy.data.clamp_(0)
            else:
                model.readout.Wx.data.clamp_(0)
                model.readout.Wy.data.clamp_(0) 
        train_loss += loss.item()
        del loss
    train_loss /= dl_length
    return train_loss

def train(model, train_dl, val_dl, train_dl_length, val_dl_length, n_neurons, batch_size,l2_readout=0.1, hs_readout=0, clamp=True, device='cuda', n_epochs_period=[100, 30, 30, 30], patience=5):
    import time
    # batch_size = 100
    detach_core = False

    n_periods = 4
    epochs_since_best = 0

    for i_period in range(n_periods):
        lr = 1e-3 / (3 ** (i_period))
        print(f'Learning rate = {lr}')

        restore = (i_period > 0)
        if restore:
            model.load_state_dict(best_state_dict)
        else:
            varexp_max = -np.inf

        tic = time.time()
        
        n_epochs = n_epochs_period[i_period]

        optimizer = torch.optim.AdamW([{'params': model.core.parameters(), 'weight_decay': 0.1},
                                    {'params': [model.readout.Wy, model.readout.Wx], 
                                        'weight_decay': 1.0},
                                    {'params': model.readout.Wc, 'weight_decay': l2_readout},
                                    {'params': model.readout.bias, 'weight_decay': 0}
                                    ], lr=lr)

        for epoch in range(n_epochs):
            model.train()
            train_loss = train_epoch(
                                model, optimizer, train_dl, dl_length=train_dl_length, epoch=epoch,
                                device=device, detach_core=detach_core, clamp=clamp, hs_reg=hs_readout
                            )
            model.eval()
            val_loss, varexp, _ = val_epoch(model, val_dl, dl_length=val_dl_length, 
                                            n_neurons=n_neurons, batch_size=batch_size, device=device)
            
            if (varexp.mean() > varexp_max) and (not np.isnan(val_loss*train_loss)):
                best_state_dict = copy_state(model)
                varexp_max = varexp.mean()
                epochs_since_best = 0
            elif np.isnan(val_loss*train_loss): # prevent overfitting
                print('nan loss')
                break
            else:
                epochs_since_best += 1

            print(f'epoch {epoch}, train_loss = {train_loss:0.4f}, val_loss = {val_loss:0.4f}, varexp_val = {varexp.mean():0.4f}, time {time.time()-tic:.2f}s')

            if epochs_since_best >= patience:
                print(f'Early stopping at epoch {epoch} due to no improvement in validation varexp.')
                break
    return best_state_dict

def monkey_val_epoch(model, img_train, spks_train, real_spks_train, batch_size=100, device = torch.device('cuda'), \
                       detach_core=False, hs_reg=0, epsilon = 1e-6):
    model.eval()
    train_loss = 0
    n_train, n_neurons = spks_train.shape
    spks_train_pred = torch.zeros((n_train, n_neurons), device=device)
    with torch.no_grad():
        for k in np.arange(0, n_train, batch_size):
            kend = min(k+batch_size, n_train)
            res_batch = spks_train[k:kend].to(device)
            imgs_batch = img_train[k:kend].to(device)
            rresp_batch = real_spks_train[k:kend].to(device)
            spks_pred = model(imgs_batch, detach_core=detach_core)
            loss = ((spks_pred - res_batch * torch.log(spks_pred))*rresp_batch).sum(axis=0) / (rresp_batch.sum(axis=0) + epsilon)
            loss += hs_reg * model.readout.hoyer_square()
            loss = loss.mean()
            train_loss += loss.item()
            spks_train_pred[k:kend] = spks_pred
        train_loss /= n_train
    spks_train_pred = spks_train_pred.detach().cpu().numpy()
    spks_train_nan = np.where(real_spks_train, spks_train, np.nan)
    varexp = np.nansum((spks_train_nan - spks_train_pred)**2, axis=0)
    spks_train -= np.nanmean(spks_train, axis=0)
    denom = np.nansum(spks_train**2, axis=0)
    varexp /= np.nansum(spks_train**2, axis=0)
    varexp = 1 - varexp
    varexp[denom == 0] = 0
    return train_loss, varexp

def monkey_train_epoch(model, optimizer, img_train, spks_train, real_spks_train, epoch=0, batch_size=100, device = torch.device('cuda'), \
                       detach_core=False, hs_reg=0, epsilon = 1e-6):
    n_train = img_train.shape[0]
    np.random.seed(epoch)
    iperm = np.random.permutation(n_train)
    train_loss = 0
    for k in np.arange(0, n_train, batch_size):
        optimizer.zero_grad()
        kend = min(k+batch_size, n_train)
        res_batch = spks_train[iperm[k:kend]].to(device)
        imgs_batch = img_train[iperm[k:kend]].to(device)
        rresp_batch = real_spks_train[iperm[k:kend]].to(device)
        spks_pred = model(imgs_batch, detach_core=detach_core)
        loss = ((spks_pred - res_batch * torch.log(spks_pred))*rresp_batch).sum(axis=0) / (rresp_batch.sum(axis=0) + epsilon)
        loss += hs_reg * model.readout.hoyer_square()
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        model.readout.Wx.data.clamp_(0)
        model.readout.Wy.data.clamp_(0) 
        train_loss += loss.item()
    train_loss /= n_train
    return train_loss

def monkey_train(model, train_responses, train_real_responses, val_responses, val_real_responses, train_images, val_images, \
                 hs_readout=0, l2_readout=0.1, device='cuda', weight_decay_core=0.1, n_epochs_period=[100, 30, 30, 30], lr_init=1e-3, batch_size=100):
    detach_core = False
    n_periods = 4
    
    for i_period in range(n_periods):
        lr = lr_init / (3 ** (i_period))
        print(lr)

        restore = (i_period > 0)
        if restore:
            model.load_state_dict(best_state_dict)
        else:
            varexp_max = -np.inf

        tic = time.time()
        
        n_epochs = n_epochs_period[i_period]

        optimizer = torch.optim.AdamW([{'params': model.core.parameters(), 'weight_decay': weight_decay_core},
                                    {'params': [model.readout.Wy, model.readout.Wx], 
                                        'weight_decay': 1.0},
                                    {'params': model.readout.Wc, 'weight_decay': l2_readout},
                                    {'params': model.readout.bias, 'weight_decay': 0}
                                    ], lr=lr)

        for epoch in range(n_epochs):
            model.train()
            train_loss = monkey_train_epoch(model, optimizer, train_images, train_responses, train_real_responses, epoch=epoch, batch_size=batch_size, device=device, detach_core=detach_core, hs_reg=hs_readout)

            model.eval()
            val_loss, varexp = monkey_val_epoch(model, val_images, val_responses, val_real_responses, batch_size=batch_size, device=device)

            if (varexp.mean() > varexp_max) and (not np.isnan(val_loss*train_loss)):
                best_state_dict = copy_state(model)
                varexp_max = varexp.mean()
            elif np.isnan(val_loss*train_loss): # prevent overfitting
                print('nan loss')
                break

            if epoch%5==0 or epoch+1==n_epochs:
                print(f'epoch {epoch}, train_loss = {train_loss:0.4f}, val_loss = {val_loss:0.4f}, varexp_val = {varexp.mean():0.4f}, time {time.time()-tic:.2f}s')
    return best_state_dict


def count_samples(dl):
    """
    Counts all the samples in an experanto dataloader
    
    :param dl: experanto dataloader
    """
    n = 0
    for _, batch in dl:
        n += batch["responses"].shape[0]
    return n


def build_img_test_and_spks_rep_all(test_dl, device="cpu"):
    """
    Takes in a test_dl and gives back the structured np.arrays 
    needed to compute the fev and feve scores
    
    :param test_dl: experanto dataloader for the test data
    :param device: 
    """
    reps = defaultdict(list)      # image_id -> list of (n_neurons,) numpy arrays
    img_by_id = {}                # image_id -> img , torch tensor (1,66,130)

    for _, batch in test_dl:
        spks_batch = batch["responses"]     # (B,1,N)
        img_batch = batch["screen"]         # (B,66,1,130) or similar
        img_ids = batch["image_id"]          # (B,)

        # ---- to CPU for storage ----
        spks_batch = spks_batch.squeeze().detach().cpu().numpy()    # (B,1,N) -> (B,N)
        
        img_batch = img_batch.squeeze().unsqueeze(1)    # (B,66,1,130) -> (B,1,66,130)
        img_batch = img_batch.detach().cpu()
        
        img_ids = img_ids.detach().cpu().numpy()

        # ---- group by image_id ----
        for i in range(len(img_ids)):
            image_id = int(img_ids[i])
            reps[image_id].append(spks_batch[i])

            # store first occurrence of the image
            if image_id not in img_by_id:
                img_by_id[image_id] = img_batch[i]  # (1,66,130)

    # stable ordering
    unique_ids = sorted(reps.keys())

    # img_test: (n_unique,1,66,130)
    img_test = torch.stack([img_by_id[k] for k in unique_ids], dim=0).to(device)

    # spks_rep_all
    spks_rep_all = np.empty(len(unique_ids), dtype=object)  # np.array (n_unique, )
    for i, image_id in enumerate(unique_ids):
        spks_rep_all[i] = np.stack(reps[image_id], axis=0)  # one sample of np.array (n_repeats, n_neurons)

    return img_test, spks_rep_all, unique_ids



