import torch
from torch.utils.data import DataLoader
from datetime import datetime

from dataset_class_nn import npz_class
from neural_net_class import AE_net, CL_maxPool, AE_CNN_maxPool2
from trainer_class import Trainer
from num_features import cn_enc_out_shape

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    folder = '30000_samples_1_months'
    dir_path = '/sps/lisaf/lkarda/H_matrices_td/' + folder + '/tfm_singles/'
    vali_path = dir_path + 'tfm_singles_validation'
    dataset = npz_class(dirpath=dir_path)
    vali_set = npz_class(dirpath=vali_path)

    epochs = 100
    batch_size = 100
    num_workers = 4
    channel_mult = 8
    

    h_cn_enc_out, w_cn_enc_out, h_kernel, w_kernel = cn_enc_out_shape(channel_mult= channel_mult, input_shape= dataset[0][0].shape)

    num_fc_nodes_after_conv= batch_size * h_cn_enc_out * w_cn_enc_out
    num_fc_nodes_bottleneck= 5000

    auto_enc = AE_CNN_maxPool2(
        # channel_mult= channel_mult,
        # h_kernel= h_kernel,
        # w_kernel= w_kernel,
        # num_fc_nodes_after_conv= num_fc_nodes_after_conv,
        # num_fc_nodes_bottleneck= num_fc_nodes_bottleneck
    ).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(auto_enc.parameters(), lr=1.e-3, weight_decay=1.e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(auto_enc.__class__.__name__)
    print('# of epochs:', epochs)
    print('# of samples:', len(dataset))
    print('Batchsize:', batch_size)
    print('# of batches:', len(dataset) // batch_size)
    print('# of nodes in bottleneck', num_fc_nodes_bottleneck)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(vali_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(timestamp)

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
        model_path = '/sps/lisaf/lkarda/model_states/{}_{}_{}_{}_{}_{}'.format(auto_enc.__class__.__name__, folder, channel_mult, num_fc_nodes_bottleneck, timestamp, epoch)
        torch.save(auto_enc.state_dict(), model_path)
        scheduler.step()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
