import ast
import torch, torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .vit_2drope_open_clip import MLCDAnyResConfig,MLCDAnyRes
import pdb


def feature_select_multi(model_outputs, select_layers=None):
    assert len(select_layers) > 0, 'selected_layers should not be empty'
    image_feature_list = [model_outputs[l][:, 1:] for l in select_layers]
    image_features_multi = torch.cat(image_feature_list, dim=2)
    return image_features_multi

class MLCDAnyResVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        tokenpack_vision_layers = getattr(args, 'tokenpack_vision_layers', None)
        self.selected_layers = None if tokenpack_vision_layers is None else \
            ast.literal_eval(tokenpack_vision_layers)
        if not delay_load:
            #here
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            #for test
            if "MLCDAnyRes" in self.vision_tower_name:
                self.vision_tower  =MLCDAnyRes.from_pretrained(self.vision_tower_name)
            else:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
                self.vision_tower = CLIPVisionModel._from_config(self.cfg_only)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)
            self.cfg_only = self.vision_tower.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        #
        if not hasattr(self, 'image_processor'):
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        
        if "MLCDAnyRes" in self.vision_tower_name:
            self.vision_tower =MLCDAnyRes.from_pretrained(self.vision_tower_name,device_map=device_map)
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            multilayer_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
                if self.selected_layers is not None:
                    multilayer_feature = feature_select_multi(image_forward_outs, select_layers=self.selected_layers).to(images.dtype)
                    multilayer_features.append(multilayer_feature)
                else:
                    multilayer_features = None
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            if self.selected_layers is not None:
                multilayer_features = feature_select_multi(image_forward_outs, select_layers=self.selected_layers).to(images.dtype)
            else:
                multilayer_features = None

        if multilayer_features is not None:
            return image_features, multilayer_features

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



