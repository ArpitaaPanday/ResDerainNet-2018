# ResDerainNet-2018 — PyTorch Implementation

A student implementation and demonstration of **“Single-Image Rain Removal Using Residual Deep Learning”** (Takuro Matsui et al., IEEE ICIP 2018).

## Repository contents
- `ResDerainNet_Group29_Implementation_.ipynb`: Google Colab notebook used for implementation and evaluation.
- `model.py`: PyTorch definition of the 20-convolution network.
- `weight_conversion.py`: helper to convert the authors’ Caffe HDF5 checkpoint to a PyTorch state dictionary.
- `requirements.txt`: Python dependencies.
- `results/`: evaluation tables extracted from saved notebook outputs.

## Method
The network predicts a rain residual. The restored image is produced by subtracting the estimated residual from the rainy input. Inference is performed on overlapping 128×128 patches (stride 106), with overlapping predictions averaged.

## Important implementation note
This is an inference/reproduction exercise using the authors’ pretrained checkpoint, not training from scratch. The notebook uploaded here visibly loads convolution weights, but does not include the BatchNorm running-statistics and Scale-parameter loading step. Therefore, do not claim exact equivalence to the original Caffe model based on this notebook alone. The standalone conversion script includes BatchNorm/Scale loading, but its HDF5 group names must be verified against the exact checkpoint before use. The CSV metrics are reported as saved notebook outputs, not proof of exact Caffe/PyTorch equivalence.

## Original authors’ code and checkpoint
Official repository: https://github.com/takuro-matsui/ResDerainNet

Obtain the pretrained checkpoint from the original authors’ repository and follow its license/usage terms. The checkpoint is intentionally not included here.

## Setup
```bash
pip install -r requirements.txt
python weight_conversion.py --checkpoint /path/to/ResDerainNet_iter_100000.caffemodel.h5
```

The notebook also expects the original repository's image assets and checkpoint in the Colab runtime.

## Results
The CSV files in `results/` contain the four-image synthetic evaluation tables extracted from the notebook's saved outputs. Add the generated comparison figures and real-world restored images from the Colab output folder if you want to publish those visuals too.

## Attribution
Original research and Caffe implementation: Takuro Matsui et al., “Single-Image Rain Removal Using Residual Deep Learning,” IEEE International Conference on Image Processing (ICIP), 2018, pp. 3928–3932.
