#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from dataclasses import dataclass

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig

def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss

def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)

@dataclass
class LlavaOutputWithPast(CausalLMOutputWithPast):
    labels: Optional[torch.FloatTensor] = None
    
class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        grounding_enc_imgs: Optional[List[torch.FloatTensor]] = None,
        image_sam_resizes: Optional[List[torch.FloatTensor]] = None,
        original_sizes: Optional[List[torch.FloatTensor]] = None,
        masks_list: Optional[List[List[torch.FloatTensor]]] = None,
        infer: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids_ = input_ids
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                old_attention_mask,
                img_token_num,
                num_images_batch
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes
            )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                cache_position=cache_position
            )
        
        if getattr(self.config, 'sam_path', None) is not None and self.config.sam_path !="":
            device = inputs_embeds.device
            sam_image_embeddings = self.get_grounding_encoder_embs(grounding_enc_imgs)
            seg_token_mask = self.create_seg_token_mask(input_ids_, old_attention_mask, img_token_num, num_images_batch)
            seg_text_embeds_batch = self.process_hidden_states(output["hidden_states"], seg_token_mask)
            pred_masks_batch = self.generate_and_postprocess_masks(seg_text_embeds_batch, sam_image_embeddings, num_images_batch, image_sam_resizes, original_sizes)
            if infer:
                return {"output":output, "pred_masks":pred_masks_batch}
            
            mask_loss = self.compute_seg_loss(pred_masks_batch, masks_list, device)
            output['loss'] += mask_loss
        return LlavaOutputWithPast(**output)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def compute_seg_loss(self, pred_masks_batch, masks_list, device):
        mask_bce_loss = torch.tensor(0.0, device=device)
        mask_dice_loss = torch.tensor(0.0, device=device)
        num_masks = 0
        for batch_i, pred_masks in enumerate(pred_masks_batch):
            gt_masks = masks_list[batch_i].float() # masks_list batch[gts,gts,...]
            # Resize gt_mask to match pred_mask if needed
            if gt_masks.shape[0] != pred_masks.shape[0]:
                gt_masks = gt_masks[:pred_masks.shape[0]]
            assert gt_masks.shape[0] == pred_masks.shape[0], f"Shape mismatch: gt_mask {gt_masks.shape}, pred_mask {pred_masks.shape}"

            # Compute Binary Cross-Entropy Loss
            mask_bce_loss += (compute_sigmoid_cross_entropy(pred_masks, gt_masks, mask_count=gt_masks.shape[0]) *
                                gt_masks.shape[0])
            # Compute Dice Loss
            mask_dice_loss += (
                    calculate_dice_loss(pred_masks, gt_masks, mask_count=gt_masks.shape[0]) * gt_masks.shape[0])
            num_masks += gt_masks.shape[0]
        # Normalize the losses
        bce_loss_weight, dice_loss_weight = 2, 0.5
        mask_bce_loss = bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        return mask_loss

    def generate_and_postprocess_masks(self, seg_text_embeds_batch, sam_image_embeddings, num_images_batch, image_sam_resizes, original_sizes):
        assert len(seg_text_embeds_batch) == len(num_images_batch)
                
        pred_masks_batch = [] # list()
        for batch_i, seg_text_embeds in enumerate(seg_text_embeds_batch):
            num_img = max(1, num_images_batch[batch_i])

            pred_mask_  = torch.empty((0, original_sizes[batch_i][0], original_sizes[batch_i][1]), device=seg_text_embeds.device)
            for img_i in range(num_img):
                sparse_embeddings, dense_embeddings = self.model.sam.prompt_encoder(
                    points=None, boxes=None, masks=None, text_embeds=seg_text_embeds.unsqueeze(1)[img_i::num_img,:,:]
                )
                sparse_embeddings = sparse_embeddings.to(seg_text_embeds.dtype)
                
                low_res_masks, _ = self.model.sam.mask_decoder(
                    image_embeddings=sam_image_embeddings[batch_i][img_i].unsqueeze(0),
                    image_pe=self.model.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings, 
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False, )
                pred_mask = self.model.sam.postprocess_masks(
                low_res_masks, input_size=image_sam_resizes[batch_i][img_i], original_size=original_sizes[batch_i],)
                pred_mask_ = torch.cat([pred_mask_, pred_mask[:,0]], dim=0)
            pred_masks_batch.append(pred_mask_)
        return  pred_masks_batch
                
    def process_hidden_states(self, output_hidden_states, seg_token_mask):
        hidden_states_ = [self.model.text2sam_projection(output_hidden_states[-1])]
        hidden_states_ = torch.stack(hidden_states_, dim=-1).sum(dim=-1)
        seg_text_embeds_batch = []
        for i, hidden_state_ in enumerate(hidden_states_):
            # assert hidden_state_.shape[0] == seg_token_mask.shape[1], f"hidden:{hidden_state_.shape}, segtoken:{seg_token_mask.shape}"
            # seg_text_embeds_batch.append(hidden_state_[seg_token_mask[i]])
            seg_text_embeds_batch.append(hidden_state_[seg_token_mask[i][:hidden_state_.shape[0]]])
        return seg_text_embeds_batch
        
    def create_seg_token_mask(self, input_ids, attention_mask, img_token_num, num_images_batch):
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        max_len = 0
        for i, _ in enumerate(input_ids):
            max_len = max(max_len, len(input_ids[i]) + img_token_num[i] - num_images_batch[i])
            
        seg_token_mask = []
        for i, _ in enumerate(input_ids):
            mask = input_ids[i][num_images_batch[i]:] == self.seg_token_idx
            seg_token_mask.append(
                torch.cat(
                    [torch.zeros((1, img_token_num[i])).bool().cuda(), mask.unsqueeze(0), torch.zeros((1, max_len-(len(input_ids[i]) + img_token_num[i] - num_images_batch[i]))).bool().cuda()], dim=1
                )
            )
        return torch.cat(seg_token_mask, dim=0)
            
    def get_grounding_encoder_embs(self, batch_images: torch.FloatTensor):
        # with torch.no_grad():
        batch_feats = []
        for images in batch_images:
            batch_feats.append(torch.cat([self._encode_single_image(img) for img in images], dim=0))
        return batch_feats

    def _encode_single_image(self, image):
        # torch.cuda.empty_cache()
        return self.model.sam.image_encoder(image.unsqueeze(0))
 
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
