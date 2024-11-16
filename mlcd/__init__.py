from .vision_transformer import ViT_B_32_224px, ViT_L_14_336px
from .vision_transformer_2d_rope import ViT_L_14_336px, ViT_g_32_512px, ViT_g_32_anyres



def get_model_name(model_name: str):
    if model_name[:4] == "MLCD":
        model_name = model_name[5:]
    if model_name == "ViT_B_32_224px":
        return ViT_B_32_224px()
    elif model_name == "ViT_L_14_336px":
        return ViT_L_14_336px()
    elif model_name == "ViT_g_32_512px":
        return ViT_g_32_512px()
    elif model_name == "ViT_g_32_anyres":
        return ViT_g_32_anyres()
    else:
        raise ValueError("Unknown model name")
