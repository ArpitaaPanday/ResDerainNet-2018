"""Convert the authors' Caffe HDF5 checkpoint to this PyTorch model.

Expected checkpoint layout follows the authors' ResDerainNet_iter_100000
Caffe HDF5 file. This script loads convolution weights, BatchNorm statistics,
and Scale gamma/beta. Check the exact HDF5 group names if using another
checkpoint revision.
"""
import argparse
import h5py
import torch
from model import ResDerainNet

def copy_array(dst, arr):
    dst.copy_(torch.as_tensor(arr, dtype=dst.dtype))

def convert(h5_path, output_path):
    model = ResDerainNet()
    with h5py.File(h5_path, "r") as f, torch.no_grad():
        root = f["data"]
        # Convolution layers 1 and 20, plus intermediate conv2..conv19.
        for layer_num in range(1, 21):
            layer = (model.conv1 if layer_num == 1 else
                     model.conv20 if layer_num == 20 else
                     model.convs[layer_num - 2])
            group = root[f"conv{layer_num}"]
            copy_array(layer.weight, group["0"][:])
            copy_array(layer.bias, group["1"][:])

        # Caffe BatchNorm stores mean, variance, and a scale factor.
        # PyTorch running statistics are the normalized mean/variance.
        for layer_num in range(2, 20):
            idx = layer_num - 2
            bn_group = root[f"conv{layer_num}-bn"]
            mean = bn_group["0"][:]
            variance = bn_group["1"][:]
            factor = float(bn_group["2"][0])
            if factor != 0:
                mean = mean / factor
                variance = variance / factor
            copy_array(model.bns[idx].running_mean, mean)
            copy_array(model.bns[idx].running_var, variance)

            # Caffe Scale layer parameters: gamma and beta.
            scale_group = root[f"conv{layer_num}-scale"]
            copy_array(model.bns[idx].weight, scale_group["0"][:])
            copy_array(model.bns[idx].bias, scale_group["1"][:])

    model.eval()
    torch.save(model.state_dict(), output_path)
    print(f"Saved converted PyTorch state_dict: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Original Caffe .caffemodel.h5")
    parser.add_argument("--output", default="ResDerainNet_pytorch.pth")
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
