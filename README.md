A simple common utils and models package for MRI analysis.

## Functions implemented in MengUtils

### Datasets

- ACDC: `from mengutils.acdc import LoadACDC`
- BraTS: `from mengutils.brats import LoadBraTS`
- MRBrainS: `from mengutils.mrbrains import LoadMRBrainS`
- H5 files: `from mengutils.ic_data import LoadH5`
- Other `*.png` datasets: `from mengutils.pngs import LoadPNGS`

### Load and save Files

- `*.npy`: `from mengutils.tonpy import SaveDataset`
- `*.nii`/`*.nii.gz`: `from mengutils.tonii import SaveNiiFile`

### Models

- 2D-UNet: `from mengutils.unet import UNet`
- 3D-UNet: `from mengutils.unet_3d import UNet3D`
- 3D-UNet with Attention: `from mengutils.unet_3d_atten import UNet3D_Atten`

### Metrics

- MRI Metrics: `from mengutils.metrics import Metrics`

### Others

- Normalization: `from mengutils.norm import Normalization`
- Time related: `from mengutils.timer import Timer`
- Print logs: `from mengutils.logs import Logs`
- Plot lines: `from mengutils.plots import Plots`

