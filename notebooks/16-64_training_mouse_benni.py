import os
import numpy as np
import torch
import argparse
from minimodel import data
from minimodel import model_builder
from minimodel import model_trainer
from minimodel import metrics



def main():
    # args parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_id", type=int)
    args = parser.parse_args()


    # setup
    device = torch.device('cuda')
    mouse_id = args.mouse_id
    data_path = '../data'
    weight_path = './checkpoints_16-64'
    pretrained_weight_path = './checkpoints_16-320'
    results_path = './results_16-64'
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    np.random.seed(1)

    # load images
    img = data.load_images(data_path, mouse_id, file=data.img_file_name[mouse_id])

    # load neurons
    fname = '%s_nat60k_%s.npz'%(data.db[mouse_id]['mname'], data.db[mouse_id]['datexp'])
    spks, istim_train, istim_test, xpos, ypos, spks_rep_all = data.load_neurons(file_path = os.path.join(data_path, fname), mouse_id = mouse_id)
    n_stim, n_neurons = spks.shape

    # split train and validation set
    itrain, ival = data.split_train_val(istim_train, train_frac=0.9)

    # normalize data
    spks, spks_rep_all = data.normalize_spks(spks, spks_rep_all, itrain)


    ineur = np.arange(0, n_neurons) #np.arange(0, n_neurons, 5)
    spks_train = torch.from_numpy(spks[itrain][:,ineur]).to(device)
    spks_val = torch.from_numpy(spks[ival][:,ineur]).to(device)

    print('spks_train: ', spks_train.shape, spks_train.min(), spks_train.max())
    print('spks_val: ', spks_val.shape, spks_val.min(), spks_val.max())

    img_train = torch.from_numpy(img[istim_train][itrain]).to(device).unsqueeze(1) # change :130 to 25:100 
    img_val = torch.from_numpy(img[istim_train][ival]).to(device).unsqueeze(1)
    img_test = torch.from_numpy(img[istim_test]).to(device).unsqueeze(1)

    print('img_train: ', img_train.shape, img_train.min(), img_train.max())
    print('img_val: ', img_val.shape, img_val.min(), img_val.max())
    print('img_test: ', img_test.shape, img_test.min(), img_test.max())

    input_Ly, input_Lx = img_train.shape[-2:]


    # Filtering only neurons which have a high enough threshold of FEV. 
    test_fev= metrics.fev(spks_rep_all)

    threshold = 0.15
    print(f'filtering neurons with FEV > {threshold}')
    valid_idxes_neurons = np.where(test_fev > threshold)[0]
    print(f'valid neurons: {len(valid_idxes_neurons)} / {len(test_fev)}')

    # We only subsample up to 100 neurons to reduce computing time to ~ 7h per mouse
    n_selecting = min(2, len(valid_idxes_neurons))
    selected_idxes_neurons = np.random.choice(valid_idxes_neurons, size=n_selecting, replace=False)

    seed = 1
    FEVE_scores = []
    FEV_scores = []
    # Building Model
    for i_neuron in selected_idxes_neurons:
        # We only train models on neurons with a test_fev >= 0.15. 
        ineur = [i_neuron]
        spks_train_one_neuron = spks_train[:, ineur]
        spks_val_one_neuron = spks_val[:, ineur]

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
            best_state_dict = model_trainer.train(model, spks_train_one_neuron, spks_val_one_neuron, img_train, img_val, device=device, l2_readout=0.2, hs_readout=hs_readout)
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