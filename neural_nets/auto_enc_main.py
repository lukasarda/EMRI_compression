import torch
from torch.utils.data import DataLoader
from datetime import datetime

from dataset_class_nn import npz_class
from neural_net_class import AE_CNN_maxPool3
from trainer_class import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    epochs = 100
    batch_size = 50
    num_workers = 4


    auto_enc = AE_CNN_maxPool3().to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(auto_enc.parameters(), lr=1.e-3, weight_decay=1.e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    folder = '30000_samples'
    dir_path = '/sps/lisaf/lkarda/H_matrices_td/' + folder + '/tfm_singles/'
    vali_path = dir_path + 'tfm_singles_validation'
    dataset = npz_class(dirpath=dir_path)
    vali_set = npz_class(dirpath=vali_path)

    print(auto_enc.__class__.__name__)
    print('# of epochs:', epochs)
    print('# of samples:', len(dataset))
    print('Batchsize:', batch_size)
    print('# of batches:', len(dataset) // batch_size)


    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(vali_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    trainer = Trainer(
        auto_enc,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        device
    )


    for epoch in range(epochs):
        print('Epoch {}:'.format(epoch + 1))
        trainer.train_one_epoch(epoch)
        trainer.validate(epoch)
        model_path = '/pbs/home/l/lkarda/EMRI_compression/neural_nets/model_states/{}_{}_{}'.format(auto_enc.__class__.__name__, timestamp, epoch)
        torch.save(auto_enc.state_dict(), model_path)
        scheduler.step()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
