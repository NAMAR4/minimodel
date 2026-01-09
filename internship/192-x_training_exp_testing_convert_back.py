import os
from collections import defaultdict
import numpy as np
import torch
import torchvision
import argparse
from minimodel import data
from minimodel import model_builder
from minimodel import model_trainer
from minimodel import model_trainer_exp
from minimodel import metrics
from pathlib import Path

from tqdm import tqdm
from omegaconf import OmegaConf, open_dict

from experanto.datasets import ChunkDataset
from experanto.dataloaders import get_multisession_dataloader



def main():
    # args parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_id", type=int)
    args = parser.parse_args()


    # setup
    device = torch.device('cuda')
    mouse_id = args.mouse_id
    weight_path = './checkpoints_192-x_exp_test'
    results_path = './results_192-x_exp_test'
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    path_to_data = '/mnt/vast-nhr/projects/bthesis_cidas_richter/benjamin/minimodel/internship/data_experanto_normalized'
    data_folder = f'nat30k_{data.mouse_names[mouse_id]}_{data.exp_date[mouse_id]}_experanto'
    data_path = os.path.join(path_to_data, data_folder)

    # print information
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("torchvision:", torchvision.__version__)
    print("cuda available:", torch.cuda.is_available())

    # load configs for dataloaders
    config_path = "./cfg_experanto/do_nothing_config.yaml"
    cfg_train = OmegaConf.load(config_path)
    cfg_val = OmegaConf.load(config_path)
    cfg_test = OmegaConf.load(config_path)
    print("Loaded experanto configfrom: ", config_path)

    cfg_train.dataset.modality_config.screen.valid_condition = {"tier": "train"}
    cfg_val.dataset.modality_config.screen.valid_condition = {"tier": "validation"}

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
    else:                               test_dl_length = model_trainer_exp.count_samples(test_dl)

    print("length of train_dl: ", train_dl_length)
    print("length of val_dl: ", val_dl_length)
    print("length of test_dl: ", test_dl_length)

    
    _ ,batch = next(iter(val_dl))
    NN = batch["responses"].shape[-1]       # number of neurons
    batch_size = cfg_val.dataloader.batch_size  # nur im val_epoch benötigt
    print("number of neurons: ", NN)
    print("Batch Size: ", batch_size)
    val_dl = get_multisession_dataloader(paths, cfg_val)    # reloading val_dl. somehow next(iter(val_dl)) destroys first batch also for new instances

    # ---- Converting back to minimodel format ----
    batch_size = cfg_train.dataloader.batch_size

    spks_train_exp = torch.zeros((train_dl_length, NN), device=device)
    img_train_exp = torch.zeros((train_dl_length, 1, 66, 130), device=device)
    index_array = np.arange(0, train_dl_length, batch_size)
    for k , (_, batch) in zip(index_array, train_dl):
        spks_batch = batch["responses"]
        img_batch = batch["screen"]
        kend = min(k+batch_size, train_dl_length)
        spks_train_exp[k:kend] = spks_batch.squeeze()
        img_batch = img_batch.squeeze().unsqueeze(1)        # shape: (batch_size, 1, 66,130)
        img_train_exp[k:kend] = img_batch


    spks_val_exp = torch.zeros((val_dl_length, NN), device=device)
    img_val_exp = torch.zeros((val_dl_length, 1, 66, 130), device=device)
    index_array = np.arange(0, val_dl_length, batch_size)
    for k , (_, batch) in zip(index_array, val_dl):
        spks_batch = batch["responses"]
        img_batch = batch["screen"]
        kend = min(k+batch_size, val_dl_length)
        spks_val_exp[k:kend] = spks_batch.squeeze()
        img_batch = img_batch.squeeze().unsqueeze(1)        # shape: (batch_size, 1, 66,130)
        img_val_exp[k:kend] = img_batch

    seed = 1
    feve_nlayers = []
    for nlayers in range(1, 5):
        # Building Model

        nconv1 = 192
        nconv2 = 192
        model, in_channels = model_builder.build_model(NN=NN, n_layers=nlayers, n_conv=nconv1, n_conv_mid=nconv2)
        model_name = model_builder.create_model_name(data.mouse_names[mouse_id], data.exp_date[mouse_id], n_layers=nlayers, in_channels=in_channels, seed=seed)
        
        model_path = os.path.join(weight_path, model_name)
        print('model path: ', model_path)
        model = model.to(device)


        # Training the model
        print(device)
        if not os.path.exists(model_path):
            best_state_dict = model_trainer.train(model, spks_train_exp, spks_val_exp, img_train_exp, img_val_exp, device=device)
            torch.save(best_state_dict, model_path)
            print('saved model', model_path)
        model.load_state_dict(torch.load(model_path))
        print('loaded model', model_path)

        # test model
        img_test, spks_rep_all, unique_ids = model_trainer_exp.build_img_test_and_spks_rep_all(test_dl, device=device)
        print("Total test images used: ", len(unique_ids))
        print("img_test: ", img_test.shape)
        print("spks_rep_all: ", spks_rep_all.shape)

        test_pred = model_trainer.test_epoch(model, img_test)
        print('test_pred: ', test_pred.shape, test_pred.min(), test_pred.max())


        test_fev, test_feve = metrics.feve(spks_rep_all, test_pred)
        print('FEVE (test, all): ', np.mean(test_feve))

        threshold = 0.15
        print(f'filtering neurons with FEV > {threshold}')
        valid_idxes = np.where(test_fev > threshold)[0]
        print(f'valid neurons: {len(valid_idxes)} / {len(test_fev)}')
        print(f'FEVE (test, FEV>0.15): {np.mean(test_feve[test_fev > threshold])}')
        feve_nlayers.append(np.mean(test_feve[test_fev > threshold]))

    
    
    # ---- Saving performance scores ----
    file_name = "results_" + str(mouse_id)
    results_file_path = os.path.join(results_path, file_name)

    feve_nlayers = np.array(feve_nlayers)
    print("saving array of shape: ", feve_nlayers.shape)
    np.savez(results_file_path, FEVE_scores=feve_nlayers)
    print(f"Results saved at: {results_file_path}")

if __name__ == "__main__":
    main()