# Blender_StyleGAN2-ada_addon
 Implementation of NVLab's StyleGAN2-ada for material creation in Blender


https://user-images.githubusercontent.com/31198560/133867325-bd95befb-e25b-472f-8138-3e102dd39c80.mp4

https://user-images.githubusercontent.com/31198560/133943020-3d2cdf14-d52f-47f1-946a-ea225afdf3a4.mp4

https://user-images.githubusercontent.com/31198560/135276267-6d149129-6099-4344-bb16-22d7b00e891c.mp4



# Setup
In order to install the needed modules, it must be done so for the Blender's included python binary
```
cd {Blender instalation path}/{Blender version}/python/bin

python.exe -m ensurepip

python -m pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
```

Install pytorch following their instructions https://pytorch.org/get-started/locally/

*Note: if you already have installed some or any of these packages, uninstall them first as they won't be installed inside Blender libraries*


Download https://github.com/NVlabs/stylegan2-ada-pytorch

Copy `legacy.py` inside `{Blender instalation path}/{Blender.version}/python/lib`

Copy `dnnlib` folder inside `{Blender instalation path}/{Blender.version}/python/lib/site-packages`

Put your trained models inside `Blender Foundation/models`

Done!



(Superresolution WIP) https://github.com/xinntao/Real-ESRGAN

Download model "RealESRGAN_x4plus.pth" and place in same directory as the other models.

```
python -m pip install basicsr, facexlib
```

Download repo and copy 'realesrgan/' to `{Blender instalation path}/{Blender.version}/python/lib/`. And for some reason I had to delete line 6 in 'realesrgan/__ init ___.py' for it to work.


# Usage (for now)
Download or train a model (lots of them here https://github.com/justinpinkney/awesome-pretrained-stylegan2)

Open the .blend

Run the script, on the 3D View sidebar, a tab named 'StyleGAN' should appear

Pick trained model .pkl (should be on the same drive as Blender installation)

With the object selected, pick a random seed and click on 'Generate Image'

To animate, simply insert keyframes for the weight value, set render path and click animate (interface will freeze until all frames are rendered. To interrupt press ctrl+c in system console).

# TODO
• Set/animate multiple weights at a time.

• Render the native way instead of dedicated render button.

• Generate image sequence for faster re-rendering


# Acknowledgements
```
@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```
