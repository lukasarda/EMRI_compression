import matplotlib.pyplot as plt

def read_log_file(file_path):
    training_losses = []
    validation_losses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('Training Loss:'):
                training_loss = float(line.split(':')[-1].strip())
                training_losses.append(training_loss)
            elif line.startswith('Validation Loss:'):
                validation_loss = float(line.split(':')[-1].strip())
                validation_losses.append(validation_loss)
    return training_losses, validation_losses

def plot_losses(training_losses, validation_losses):
    epochs = range(1, len(training_losses) + 1)
    plt.semilogy(epochs, training_losses, label='Training Loss')
    plt.semilogy(epochs, validation_losses, label='Validation Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/pbs/home/l/lkarda/EMRI_compression/neural_nets/loss/loss_AE_CNN_maxPool2_new_norm.png')

if __name__ == "__main__":
    log_file_path = "/pbs/home/l/lkarda/EMRI_compression/serial_test_15917239.log"  # Replace with the actual path to your log file
    training_losses, validation_losses = read_log_file(log_file_path)
    plot_losses(training_losses, validation_losses)
