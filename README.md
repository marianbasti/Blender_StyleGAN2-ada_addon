# Blender_StyleGAN2-ada_addon
 Implementation of NVLab's StyleGAN2-ada for material creation in Blender

# Setup
In order to install the needed modules, it must be done so for the Blender's inculded python binary

`cd Blender/{Blender.version}/python/bin`

`pip ensurepip`


`pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`

Also install pytorch following their instructions https://pytorch.org/get-started/locally/

*Note: if you already have installed some or any of these packages, uninstall them first as they won't be installed inside Blender libraries*


Download https://github.com/NVlabs/stylegan2-ada-pytorch
Copy `legacy.py` inside `Blender/{Blender.version}/python/lib`
Copy `dnnlib` folder inside `Blender/{Blender.version}/python/lib/site-packages`

Done!
