import os
import numpy as np
import torch
import gc


from minimodel import data
from minimodel import model_trainer_exp

import matplotlib.pyplot as plt
from omegaconf import OmegaConf, open_dict
from experanto.datasets import ChunkDataset
from experanto.dataloaders import get_multisession_dataloader





def main():
    # minimodel

    # setup
    device = "cpu"
    mouse_id = 0
    data_path = '../data'
    np.random.seed(1)

    # load images
    img = data.load_images(data_path, mouse_id, file=data.img_file_name[mouse_id])

    # load neurons
    fname = '%s_nat60k_%s.npz'%(data.db[mouse_id]['mname'], data.db[mouse_id]['datexp'])
    spks, istim_train, istim_test, xpos, ypos, spks_rep_all = data.load_neurons(file_path = os.path.join(data_path, fname), mouse_id = mouse_id)
    n_stim, n_neurons = spks.shape
    print("spks_rep_all: ", spks_rep_all.shape)
    print("spks: ", spks.shape)
    # split train and validation set
    itrain, ival = data.split_train_val(istim_train, train_frac=0.9)

    # normalize data
    spks, spks_rep_all = data.normalize_spks(spks, spks_rep_all, itrain)


    ineur = np.arange(0, n_neurons) #np.arange(0, n_neurons, 5)
    spks_train = torch.from_numpy(spks[itrain][:,ineur]).to(device)
    spks_val = torch.from_numpy(spks[ival][:,ineur]).to(device)

    print('spks_train: ', spks_train.shape, spks_train.min(), spks_train.max())
    print('spks_val: ', spks_val.shape, spks_val.min(), spks_val.max())
    print('spks_test: ', spks_rep_all.shape, " with shape of a sample: ", spks_rep_all[0].shape)

    img_train = torch.from_numpy(img[istim_train][itrain]).to(device).unsqueeze(1) # change :130 to 25:100 
    img_val = torch.from_numpy(img[istim_train][ival]).to(device).unsqueeze(1)
    img_test = img[istim_test]

    print('img_train: ', img_train.shape, img_train.min(), img_train.max())
    print('img_val: ', img_val.shape, img_val.min(), img_val.max())
    print('img_test: ', img_test.shape, img_test.min(), img_test.max())
    # experanto

    # --- setup ---
    #path_to_data = '/mnt/vast-nhr/projects/bthesis_cidas_richter/benjamin/minimodel/internship/data_experanto_normalized'
    path_to_data = '/mnt/vast-nhr/projects/bthesis_cidas_richter/benjamin/minimodel/internship/data_experanto'
    data_folder = f'nat30k_{data.mouse_names[mouse_id]}_{data.exp_date[mouse_id]}_experanto'
    data_path = os.path.join(path_to_data, data_folder)

    # load configs for dataloaders
    #cfg_train = OmegaConf.load("./cfg_experanto/do_nothing_config.yaml")
    #cfg_val = OmegaConf.load("./cfg_experanto/do_nothing_config.yaml")
    #cfg_test = OmegaConf.load("./cfg_experanto/do_nothing_config.yaml")
    cfg_train = OmegaConf.load("./cfg_experanto/basic_config.yaml")
    cfg_val = OmegaConf.load("./cfg_experanto/basic_config.yaml")
    cfg_test = OmegaConf.load("./cfg_experanto/basic_config.yaml")

    cfg_train.dataset.modality_config.screen.valid_condition = {"tier": "train"}
    cfg_train.dataset.out_keys.append("image_id")             # using this for debugging purposes
    cfg_train.dataloader.shuffle = False                     # using this for debugging purposes
    cfg_train.dataloader.drop_last = False

    cfg_val.dataset.modality_config.screen.valid_condition = {"tier": "validation"}
    cfg_val.dataset.out_keys.append("image_id")             # using this for debugging purposes
    cfg_val.dataloader.shuffle = False                      # using this for debugging purposes
    cfg_val.dataloader.drop_last = False

    cfg_test.dataset.modality_config.screen.valid_condition = {"tier": "test"}
    cfg_test.dataset.out_keys.append("image_id")        # I sadly need this to combine all samples with same image_id
    cfg_test.dataloader.drop_last = False               # Here I dont need the batches to be the same size
    cfg_test.dataloader.shuffle = False

    # build dataloaders
    paths = [data_path]
    train_dl = get_multisession_dataloader(paths, cfg_train)
    val_dl = get_multisession_dataloader(paths, cfg_val)
    test_dl = get_multisession_dataloader(paths, cfg_test)
    print("Loaded experanto data from: ", data_path)

    if cfg_train.dataloader.drop_last:  train_dl_length = len(train_dl) * cfg_train.dataloader.batch_size
    else:                               train_dl_length = model_trainer_exp.count_samples(train_dl)
    if cfg_val.dataloader.drop_last:    val_dl_length = len(val_dl) * cfg_val.dataloader.batch_size
    else:                               val_dl_length = model_trainer_exp.count_samples(val_dl)
    if cfg_test.dataloader.drop_last:   test_dl_length = len(test_dl) * cfg_test.dataloader.batch_size
    else:                               test_dl_length =model_trainer_exp.count_samples(test_dl)

    print("length of train_dl: ", train_dl_length)
    print("length of val_dl: ", val_dl_length)
    print("length of test_dl: ", test_dl_length)

    batch_size = cfg_train.dataloader.batch_size

    spks_train_exp = torch.zeros((train_dl_length, n_neurons), device=device)
    img_train_exp = torch.zeros((train_dl_length, 1, 66, 130), device=device)
    index_array = np.arange(0, train_dl_length, batch_size)
    for k , (_, batch) in zip(index_array, train_dl):
        spks_batch = batch["responses"]
        img_batch = batch["screen"]
        kend = min(k+batch_size, train_dl_length)
        spks_train_exp[k:kend] = spks_batch.squeeze()
        img_batch = img_batch.squeeze().unsqueeze(1)        # shape: (batch_size, 1, 66,130)
        img_train_exp[k:kend] = img_batch



    compare_formats(spks_train_exp, spks_train, modality="spks", tier="train", only_diff=True)
    compare_formats(img_train_exp, img_train, modality="img", tier="train", only_diff=True)
    batch_size = cfg_val.dataloader.batch_size

    spks_val_exp = torch.zeros((val_dl_length, n_neurons), device=device)
    img_val_exp = torch.zeros((val_dl_length, 1, 66, 130), device=device)
    index_array = np.arange(0, val_dl_length, batch_size)
    for k , (_, batch) in zip(index_array, val_dl):
        spks_batch = batch["responses"]
        img_batch = batch["screen"]
        kend = min(k+batch_size, val_dl_length)
        spks_val_exp[k:kend] = spks_batch.squeeze()
        img_batch = img_batch.squeeze().unsqueeze(1)        # shape: (batch_size, 1, 66,130)
        img_val_exp[k:kend] = img_batch
    compare_formats(spks_val_exp, spks_val, modality="spks", tier="val", only_diff=True)
    compare_formats(img_val_exp, img_val, modality="img", tier="val", only_diff=True)
    batch_size = cfg_test.dataloader.batch_size

    spks_test_exp = torch.zeros((test_dl_length, n_neurons), device=device)
    img_test_exp = torch.zeros((test_dl_length, 1, 66, 130), device=device)
    index_array = np.arange(0, test_dl_length, batch_size)
    for k , (_, batch) in zip(index_array, test_dl):
        spks_batch = batch["responses"]
        img_batch = batch["screen"]
        kend = min(k+batch_size, test_dl_length)
        spks_test_exp[k:kend] = spks_batch.squeeze()
        img_batch = img_batch.squeeze().unsqueeze(1)        # shape: (batch_size, 1, 66,130)
        img_test_exp[k:kend] = img_batch


    # Initialize lists to store reshaped spks_test and img_test
    spks_test_list = []
    img_test_list= []
    # Iterate over each stimulus in the test set
    for i, spks in enumerate(spks_rep_all):
        nrep, nneurons = spks.shape
        spks_test_list.append(spks)
        # Reshape the corresponding img_test
        img_rep = img_test[i][None, :, :].repeat(nrep, axis=0)
        img_test_list.append(img_rep)

    spks_test = np.concatenate(spks_test_list, axis=0)
    img_test = np.concatenate(img_test_list, axis=0)

    spks_test = torch.from_numpy(spks_test[:-1])        # remove last element from test set 
    img_test = torch.from_numpy(img_test[:-1])          # remove last element from test set

    print("img_test_exp shape: ", img_test_exp.shape)
    print("img_test shape: ", img_test.shape)
    compare_formats(spks_test_exp, spks_test, modality="spks", tier="test", only_diff=True)
    print("exp:", img_test_exp.device)
    print("mini:", img_test.device)
    compare_formats(img_test_exp, img_test, modality="img", tier="test", only_diff=True)


def compare_formats(data_exp, data_mini, modality="", tier="", only_diff=False):
        """ modality is either spks or img
        """
        gc.collect()
        torch.cuda.empty_cache()
        data_exp = data_exp.to("cpu")
        data_mini = data_mini.to("cpu")

        print("=========== Now comparing data from modality:", modality, ", tier:", tier, " ===========")
        if not only_diff:
            # exp info
            print(modality+"_"+tier+"_exp: ", data_exp.shape, " min: ", data_exp.min().item(), " max: ", data_exp.max().item())
            data_exp_mean = torch.mean(data_exp)
            data_exp_std = torch.std(data_exp)
            print(modality+"_"+tier+" mean exp: ", data_exp_mean.cpu().numpy())
            print(modality+"_"+tier+" std exp: ", data_exp_std.cpu().numpy())
            print()
            # mini info:
            print(modality+"_"+tier+"_mini: ", data_mini.shape, " min: ", data_mini.min().item(), " max: ", data_mini.max().item())
            mini_spks_mean = torch.mean(data_mini)
            mini_spks_std = torch.std(data_mini)
            print(modality+"_"+tier+" mean mini: ", mini_spks_mean.cpu().numpy())
            print(modality+"_"+tier+" std mini: ", mini_spks_std.cpu().numpy())
            print()
            if modality=="spks":
                print("EXP samples: ", data_exp[:10,1])
                print("MINI samples: ", data_mini[:10,1])
            elif modality=="img":
                print("EXP samples: ", data_exp[1,0])
                print("MINI samples: ", data_mini[1,0])

        if data_exp.dtype != data_mini.dtype:
            data_exp = data_exp.to(dtype=data_mini.dtype) # dtype anpassen falls nötig

        numerical_same = torch.allclose(data_exp[:50], data_mini[:50], rtol=1e-5,atol=1e-8)  # atol: absolute tolerance, rtol relative tolerance: |a-b| <= atol + rtol*|b| for every element
        print("Tensors numericaly the same: ", numerical_same)
        

        diff_stats_dict = diff_stats(data_exp, data_mini)
        print(data_exp.shape)
        print("Total difference between " +modality+"_"+tier+" exp and " +modality+"_"+tier+" mini is: ", diff_stats_dict["diff_sum"])
        print("Mean difference per sample: ", diff_stats_dict["diff_mean"])
        print("Max difference for a sample: ", diff_stats_dict["diff_max"])

        print("Max rel diff:", diff_stats_dict["rel_diff_max"])
        print("Mean rel diff:", diff_stats_dict["rel_diff_mean"])
        print()

def diff_stats(data_exp: torch.Tensor,data_mini: torch.Tensor, chunk_elems: int = 500_000):
    """
    Computes:
      - diff.sum()
      - diff.mean()
      - diff.max()
      - rel_diff.max()
      - rel_diff.mean()

    without allocating full diff / rel_diff tensors.

    Returns a dict with the results.
    """

    # auf gemeinsamen Prefix flatten (falls Shapes minimal differieren)
    a = data_exp.reshape(-1)
    b = data_mini.reshape(-1)
    n = min(a.numel(), b.numel())

    total_abs = 0.0
    total_rel = 0.0
    max_abs = 0.0
    max_rel = 0.0

    for start in range(0, n, chunk_elems):
        end = min(start + chunk_elems, n)
        aa = a[start:end]
        bb = b[start:end]

        diff = (aa - bb).abs()   # nur Chunk-groß

        # absolute diff
        total_abs += diff.sum().item()
        max_abs = max(max_abs, diff.max().item())

        # relative diff
        rel = diff / (bb.abs() + 1e-20)
        total_rel += rel.sum().item()
        max_rel = max(max_rel, rel.max().item())

    mean_abs = total_abs / n if n > 0 else float("nan")
    mean_rel = total_rel / n if n > 0 else float("nan")

    return {
        "diff_sum": total_abs,
        "diff_mean": mean_abs,
        "diff_max": max_abs,
        "rel_diff_max": max_rel,
        "rel_diff_mean": mean_rel,
    }



if __name__ == "__main__":
    main()