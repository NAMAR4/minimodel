import os
import argparse
import numpy as np
import torch
import torchvision

from minimodel import data
from minimodel import model_builder
from minimodel import model_trainer
from minimodel import model_trainer_exp
from minimodel import metrics

from omegaconf import OmegaConf
from experanto.datasets import ChunkDataset
from experanto.dataloaders import get_multisession_dataloader

from torch.utils.data import DataLoader



def main():
    # --- args parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_id", type=int)
    args = parser.parse_args()

    # --- setup ---
    device = torch.device('cuda')
    mouse_id = args.mouse_id
    weight_path = './checkpoints_16-320_exp_test_pytorch_dl'
    results_path = './results_16-320_exp_test_pytorch_dl'
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
    cfg_train = OmegaConf.load("./cfg_experanto/do_nothing_config.yaml")
    cfg_val = OmegaConf.load("./cfg_experanto/do_nothing_config.yaml")
    cfg_test = OmegaConf.load("./cfg_experanto/do_nothing_config.yaml")

    cfg_train.dataset.modality_config.screen.valid_condition = {"tier": "train"}
    cfg_train.dataloader.batch_size = 100
    cfg_test.dataloader.shuffle = True
    cfg_val.dataset.modality_config.screen.valid_condition = {"tier": "validation"}
    cfg_val.dataloader.batch_size = 100

    cfg_test.dataset.modality_config.screen.valid_condition = {"tier": "test"}
    cfg_test.dataset.out_keys.append("image_id")        # I sadly need this to combine all samples with same image_id
    cfg_test.dataloader.drop_last = False               # Here I dont need the batches to be the same size
    cfg_test.dataloader.shuffle = False

    # build dataloaders
    dataset = ChunkDataset(data_path, **cfg_train.dataset)
    train_dl = DataLoader(dataset, **cfg_train.dataloader)

    dataset = ChunkDataset(data_path, **cfg_val.dataset)
    val_dl = DataLoader(dataset, **cfg_val.dataloader)

    paths = [data_path]
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

    
    batch = next(iter(train_dl))
    NN = batch["responses"].shape[-1]       # number of neurons
    batch_size = cfg_val.dataloader.batch_size  # nur im val_epoch benötigt
    print("number of neurons: ", NN)
    print("Batch Size: ", batch_size)


    # --- Building Model ---
    nlayers = 2
    nconv1 = 16
    nconv2 = 320
    model, in_channels = model_builder.build_model(NN=NN, n_layers=nlayers, n_conv=nconv1, n_conv_mid=nconv2)
    model_name = model_builder.create_model_name(data.mouse_names[mouse_id], data.exp_date[mouse_id], n_layers=nlayers, in_channels=in_channels)

    model_path = os.path.join(weight_path, model_name)
    print('model path: ', model_path)
    model = model.to(device)

    # --- Training the model ---
    print(device)
    if not os.path.exists(model_path):
        best_state_dict = model_trainer_exp.train(model, train_dl=train_dl, val_dl=val_dl, 
                                                train_dl_length=train_dl_length, val_dl_length=val_dl_length, 
                                                n_neurons=NN, batch_size=batch_size ,device=device)
        torch.save(best_state_dict, model_path)
        print('saved model', model_path)
    model.load_state_dict(torch.load(model_path))
    print('loaded model', model_path)

    # --- test model ---
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

    # ---- Saving performance scores ----
    file_name = "results_" + str(mouse_id)
    results_file_path = os.path.join(results_path, file_name)

    print(f"Results saved at: {results_file_path}")
    ineur = np.arange(0, NN)
    np.savez(results_file_path, FEV_scores=test_fev, FEVE_scores=test_feve, neurons_index=ineur)







if __name__ == "__main__":
    main()