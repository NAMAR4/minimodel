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
from experanto.dataloaders import get_multisession_dataloader


def main():
    # --- args parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_id", type=int)
    args = parser.parse_args()

    # --- setup ---
    device = torch.device('cuda')
    mouse_id = args.mouse_id
    pretrained_weight_path = './checkpoints_16-320_exp'  # Which should I take?
    weight_path = './checkpoints_16-64_exp'
    results_path = './results_16-64_exp'
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    path_to_data = '/mnt/vast-nhr/projects/bthesis_cidas_richter/benjamin/minimodel/internship/data_experanto'
    data_folder = f'nat30k_{data.mouse_names[mouse_id]}_{data.exp_date[mouse_id]}_experanto'
    data_path = os.path.join(path_to_data, data_folder)

    # print information
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("torchvision:", torchvision.__version__)
    print("cuda available:", torch.cuda.is_available())

    # load configs for dataloaders
    cfg_train = OmegaConf.load("./cfg_experanto/basic_config.yaml")
    cfg_val = OmegaConf.load("./cfg_experanto/basic_config.yaml")
    cfg_test = OmegaConf.load("./cfg_experanto/basic_config.yaml")

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
    else:                               test_dl_length =model_trainer_exp.count_samples(test_dl)

    print("length of train_dl: ", train_dl_length)
    print("length of val_dl: ", val_dl_length)
    print("length of test_dl: ", test_dl_length)

    
    _ ,batch = next(iter(train_dl))
    NN = batch["responses"].shape[-1]       # number of neurons
    batch_size = cfg_val.dataloader.batch_size  # nur im val_epoch benötigt
    print("number of neurons: ", NN)
    print("Batch Size: ", batch_size)

    
    # Need the data in this specific shape to use metrics. 
    # shape of spks_rep_all: (n_test_img, ) with one sample of n_test_img: (n_repeats, n_neurons)
    img_test, spks_rep_all, unique_ids = model_trainer_exp.build_img_test_and_spks_rep_all(test_dl, device=device)



    # Filtering only neurons which have a high enough threshold of FEV. 
    test_fev= metrics.fev(spks_rep_all)

    threshold = 0.15
    print(f'filtering neurons with FEV > {threshold}')
    valid_idxes_neurons = np.where(test_fev > threshold)[0]
    print(f'valid neurons: {len(valid_idxes_neurons)} / {len(test_fev)}')

    # We only subsample up to 100 neurons to reduce computing time to ~ 7h per mouse
    n_selecting = min(50, len(valid_idxes_neurons))
    seed = 1
    np.random.seed(seed)
    selected_idxes_neurons = np.random.choice(valid_idxes_neurons, size=n_selecting, replace=False)

    FEVE_scores = []
    FEV_scores = []
    # Building Model
    for i_neuron in selected_idxes_neurons:
        # We only train models on neurons with a test_fev >= 0.15. 
        ineur = [i_neuron]

        nlayers = 2
        nconv1 = 16
        nconv2 = 64
        hs_readout = 0.03
        wc_coef = 0.2
        model, in_channels = model_builder.build_model(NN=1, n_layers=nlayers, n_conv=nconv1, n_conv_mid=nconv2, Wc_coef=wc_coef)
        model_name = model_builder.create_model_name(data.mouse_names[mouse_id], data.exp_date[mouse_id], ineuron=ineur[0], n_layers=nlayers, in_channels=in_channels, seed=seed,hs_readout=hs_readout)

        model_path = os.path.join(weight_path, model_name)
        model = model.to(device)
        print('model path: ', model_path)

        # Training the model
        if not os.path.exists(model_path):
            if mouse_id == 5: pretrained_model_path = os.path.join(pretrained_weight_path, f'{data.mouse_names[mouse_id]}_{data.exp_date[mouse_id]}_2layer_16_320_clamp_norm_depthsep_pool_xrange_176.pt')
            else: pretrained_model_path = os.path.join(pretrained_weight_path, f'{data.mouse_names[mouse_id]}_{data.exp_date[mouse_id]}_2layer_16_320_clamp_norm_depthsep_pool.pt')
            print('pretrained_model_path: ', pretrained_model_path)
            pretrained_state_dict = torch.load(pretrained_model_path, map_location=device)
            # initialize conv1 with the fullmodel weights
            model.core.features.layer0.conv.weight.data = pretrained_state_dict['core.features.layer0.conv.weight']
            model.core.features.layer0.conv.weight.requires_grad = False

            best_state_dict = model_trainer_exp.train(model, train_dl=train_dl, val_dl=val_dl, 
                                                train_dl_length=train_dl_length, val_dl_length=val_dl_length, 
                                                n_neurons=NN, batch_size=batch_size, set_seed=False, mini_neuron_idx=ineur[0], device=device)
            
            torch.save(best_state_dict, model_path)
            print('saved model', model_path)
        model.load_state_dict(torch.load(model_path))
        print('loaded model', model_path)



        # test model
        test_pred = model_trainer.test_epoch(model, img_test)
        print('test_pred: ', test_pred.shape, test_pred.min(), test_pred.max())

        spks_rep = []
        for i in range(len(spks_rep_all)):
            spks_rep.append(spks_rep_all[i][:,ineur])
        test_fev, test_feve = metrics.feve(spks_rep, test_pred)

        print('FEV (test): ', np.mean(test_fev))
        print('FEVE (test): ', np.mean(test_feve))
        FEV_scores.append(test_fev)
        FEVE_scores.append(test_feve)


    file_name = "results_" + str(mouse_id)
    results_file_path = os.path.join(results_path, file_name)
    
    print(f"Results saved at: {results_file_path}")
    np.savez(results_file_path, FEV_scores=FEV_scores, FEVE_scores=FEVE_scores, neurons_index=selected_idxes_neurons)


if __name__ == "__main__":
    main()