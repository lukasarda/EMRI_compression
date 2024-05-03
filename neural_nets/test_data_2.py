import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset_class_nn import npz_class
from neural_net_class import AE_cnn2

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auto_enc = AE_cnn2().to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(auto_enc.parameters(), lr=1e-1, weight_decay=1e-8)

folder = '5020_samples'
dir_path = '/sps/lisaf/lkarda/H_matrices_td/' + folder + '/tfm_singles/'
vali_path = dir_path + 'tfm_singles_validation'

dataset = npz_class(dirpath=dir_path)
vali_set = npz_class(dirpath=vali_path)

batch_size = 20
num_workers = 4

print('# of samples:', len(dataset))
print('Batchsize:', batch_size)
print('# of batches:', len(dataset) // batch_size)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

vali_loader = torch.utils.data.DataLoader(
    vali_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

epochs = 5

for epoch_number in range(epochs):
    print('Epoch {}:'.format(epoch_number + 1))

    auto_enc.train()
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        batch = batch.float().to(device)  # Move batch to GPU if available
        optimizer.zero_grad()
        output = auto_enc(batch)
        loss = loss_function(output, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss/5
            print('batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_number * len(train_loader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    avg_loss = running_loss / len(train_loader)
    print('Training Loss:', avg_loss)
    writer.add_scalar('Loss/train', avg_loss, epoch_number)

    auto_enc.eval()
    running_vloss = 0.0

    with torch.no_grad():  # Disable gradient calculation for validation
        for vdata in vali_loader:
            vdata = vdata.float().to(device)  # Move validation data to GPU if available
            voutputs = auto_enc(vdata)
            vloss = loss_function(voutputs, vdata)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / len(vali_loader)
    print('Validation Loss:', avg_vloss)
    writer.add_scalar('Loss/validation', avg_vloss, epoch_number)

    model_path = 'AE_CNN_{}_{}'.format(timestamp, epoch_number)
    torch.save(auto_enc.state_dict(), model_path)

writer.close()
