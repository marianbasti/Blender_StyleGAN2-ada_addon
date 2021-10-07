bl_info = {
    "name": "StyleGAN for Blender",
    "version": (0, 2),
    "blender": (3, 0, 0),
    "location": "View3D > Panel > StyleGAN",
    "description": "Inference StyleGAN trained models to generate textures",
    "category": "StyleGAN",
}

import bpy
from bpy.types import Panel
from bpy.props import *
from bpy import context, data, ops
import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import glob

import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

device = torch.device('cuda')

with dnnlib.util.open_url('C:/pkl/textures.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

def pil_to_image(pil_image, name='texture'):
    '''
    PIL image pixels is 2D array of byte tuple (when mode is 'RGB', 'RGBA') or byte (when mode is 'L')
    bpy image pixels is flat array of normalized values in RGBA order
    '''
    # setup PIL image conversion
    width = pil_image.width
    height = pil_image.height
    byte_to_normalized = 1.0 / 255.0
    # create new image
    bpy_image = bpy.data.images.new(name, width=width, height=height)
    # convert Image 'L' to 'RGBA', normalize then flatten 
    bpy_image.pixels[:] = (np.asarray(pil_image.convert('RGBA'),dtype=np.float32) * byte_to_normalized).ravel()

    return bpy_image

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


#----------------------------------------------------------------------------
def generate_images(network_pkl, seeds, truncation_psi, noise_mode, vector, param, sr, srf):
    print('Loading networks from "%s"...' % network_pkl)
    #device = torch.device('cuda')
    #with dnnlib.util.open_url(network_pkl) as f:
    #    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        
    #G = torch.load(network_pkl)

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    
    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        ndarray[0,vector] = param
        z = torch.from_numpy(ndarray).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        im = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')


    if sr == True:
        #RealESRGAN
        torch.cuda.empty_cache()
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        upsampler = RealESRGANer(
            scale=4,
            model_path="C:/pkl/RealESRGAN_x4plus.pth",
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True)
        
        if srf == "2":
            im=im.resize([512,512])
        img = np.array(im)   
        h, w = img.shape[0:2]
        try:
            output, _ = upsampler.enhance(img, outscale=4)
        except Exception as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')     
        mat = bpy.context.view_layer.objects.active.active_material
        image_node = mat.node_tree.nodes["Image Texture"]
        for img in bpy.data.images:
            bpy.data.images.remove(img)
        output = PIL.Image.fromarray(output, 'RGB')
        output =  pil_to_image(output)
        image_node.image = output
    else:
        mat = bpy.context.view_layer.objects.active.active_material
        image_node = mat.node_tree.nodes["Image Texture"]
        for img in bpy.data.images:
            bpy.data.images.remove(img)
        output = pil_to_image(im)
        image_node.image = output
        torch.cuda.empty_cache()


def updateNdarray(seed):
    global ndarray
    ndarray = np.random.RandomState(seed).randn(1, G.z_dim)
#----------------------------------------------------------------------------
#Main Panel
class PANEL_PT_StyleGAN2(Panel):
    bl_label = 'StyleGAN'
    bl_space_type = 'VIEW_3D'
    bl_region_type= 'UI'
    bl_category= 'StyleGAN'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.props
        row = layout.row()
        layout.prop(props, 'network')
        row = layout.row()
        row.operator("stylegan.loadnetwork")
        layout.prop(props, 'seed')
        row = layout.row()
        layout.prop(props, 'Reseed')
        layout.prop(props, 'SuperResolution')
        layout.prop(props, 'SuperResolutionFactor')
        row = layout.row()
        layout.prop(props, 'vector')
        layout.prop(props, 'param')
        row = layout.row()
        row.operator("stylegan.run")
        row = layout.row()
        layout.prop(props, 'renderpath')
        row = layout.row()
        row.operator("stylegan.renderanim")

    
#Properties
class properties(bpy.types.PropertyGroup):
    network : StringProperty(description="Load trained model",subtype='FILE_PATH')
    seed : IntProperty(name="Seed",default = 33, min=0)
    vector : IntProperty(name="Vector",default = 0, min=1, max=512)
    param : FloatProperty(name="Value",default = 0, min=-10, max=10)
    renderpath : StringProperty(description="Render path",subtype='DIR_PATH')
    Reseed : BoolProperty(description="Regenerate weights with seed. Disable to avoid getting weights overwritten", default=True)
    SuperResolution : BoolProperty(description="Use ESRGAN for image upscale. At least 8gb VRAM needed", default=False)
    SuperResolutionFactor : EnumProperty(items=[("2","x2","x2"),("4","x4","x4")], name="Upres Factor", description="Upres Factor")

#Load .pkl
class stylegan_OT_loadNetwork(bpy.types.Operator):
    bl_label = "Load Network"
    bl_idname = "stylegan.loadnetwork"
    bl_parent_id = 'PANEL'
    bl_space_type = 'VIEW_3D'
    bl_region_type= 'UI'
    bl_category= 'StyleGAN'
    
    def execute(self,context):
        props = context.scene.props
        network_pkl = props.network
        device = torch.device('cuda')
        print('Loading %s' %network_pkl)
        global G
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        print('Success!')
        return{'FINISHED'}

# Generate Images operator
class stylegan_OT_run(bpy.types.Operator):
    bl_label = "Generate Image"
    bl_idname = "stylegan.run"
    
    def execute(self,context):
        props = context.scene.props
        if props.Reseed:
            updateNdarray(props.seed)
        generate_images(props.network, [props.seed],1,'const', props.vector, props.param, props.SuperResolution, props.SuperResolutionFactor)
        return{'FINISHED'}

#Render animation with animated parameters
class stylegan_OT_renderanim(bpy.types.Operator):
    bl_label = "Render Animation"
    bl_idname = "stylegan.renderanim"
    
    def execute(self,context):
        s=bpy.context.scene
        s.render.resolution_x = 1080 # just for my example
        s.render.resolution_y = 1080
        props = s.props
        for i in range(s.frame_start,s.frame_end):
            s.frame_current = i
            updateNdarray(props.seed)
            generate_images(props.network, [props.seed],1,'const', props.vector, props.param, props.SuperResolution, props.SuperResolutionFactor)

            s.render.filepath = (
                                props.renderpath
                                + str(s.frame_current ).zfill(3)
                                )
            bpy.ops.render.render( #{'dict': "override"},
                                  #'INVOKE_DEFAULT',  
                                  False,            # undo support
                                  animation=False, 
                                  write_still=True
                                 )


classes = (
    PANEL_PT_StyleGAN2,
    properties,
    stylegan_OT_run,
    stylegan_OT_loadNetwork,
    stylegan_OT_renderanim
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Scene.props = bpy.props.PointerProperty(type=properties)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.props
    
if __name__ == '__main__':
    register()

#---------------------------------------------------------------