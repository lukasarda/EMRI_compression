import os
import shutil

def main():
    # Directory containing .npz files
    source_dir = "/sps/lisaf/lkarda/H_matrices_td/30000_samples_1_months/tfm_singles"

    # Destination directory for the first 20% of sorted files
    dest_dir = source_dir + "/tfm_singles_validation"

    # Get list of all .npz files in the source directory
    npz_files = [f for f in os.listdir(source_dir) if f.endswith('.npz')]

    # Sort the files
    npz_files.sort()

    # Calculate the number of files to move (20%)
    num_files_to_move = int(len(npz_files) * 0.2)

    # Move the first 20% of files to the destination directory
    for i in range(num_files_to_move):
        file_to_move = npz_files[i]
        src_path = os.path.join(source_dir, file_to_move)
        dest_path = os.path.join(dest_dir, file_to_move)
        shutil.move(src_path, dest_path)
        print(f"Moved {file_to_move} to {dest_dir}")
    
if __name__ == "__main__":
    main()
