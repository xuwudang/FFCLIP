import torch
import torch.nn as nn
from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import new_class_names_coco, BACKGROUND_CATEGORY_COCO
from pytorch_grad_cam import GradCAM
from clip.clip_tool import perform_single_coco_cam, generate_cam_label, generate_clip_fts
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from FFCLIP_model.PAR import PAR
from FFCLIP_model.WFF import WFF, extract_edge_feature


def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()

def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class FFCLIP(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, dataset_root_path=None, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)
        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model)
        self.fg_text_features = zeroshot_classifier(new_class_names_coco, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)


        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)

        self.root_path = os.path.join(dataset_root_path, 'SegmentationClass')

        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        
        # Add WFF module for refinedCAM enhancement
        self.wff = WFF(channel=self.embedding_dim).cuda()
        
        self.iter_num = 0
        self.require_all_fts = True


    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups
    


    def forward(self, img, img_names, mode='train'):

        cam_list = []
        b, c, h, w = img.shape
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        fts_all_stack = torch.stack(fts_all, dim=0)  # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        if self.require_all_fts == True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3)  # (1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h // 16, w // 16)  # (11, b, c, h, w)

        fts = self.decoder_fts_fuse(all_img_tokens)
        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape

        seg, seg_attn_weight_list = self.decoder(fts)

        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        if mode=='val':
            return seg, None, attn_pred

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, 'train', str(img_name)+'.png')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

            if self.iter_num > 40000 or mode=='val': #40000
                require_seg_trans = True
            else:
                require_seg_trans = False

            cam_refined_list, keys, w, h = perform_single_coco_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   self.bg_text_features, self.fg_text_features,
                                                                   self.grad_cam,
                                                                   mode=mode,
                                                                   require_seg_trans=require_seg_trans)

            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)

            cams = cam_dict['refined_cam'].cuda()

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()
            cams = torch.cat([bg_score, cams], dim=0).cuda()

            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()

            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
                
                # Apply WFF module to enhance refinedCAM with edge features
                if mode == 'train' or self.iter_num > 20000:  # Apply WFF after some training
                    # Extract edge features from the original image
                    edge_feature = extract_edge_feature(img[i].unsqueeze(0))
                    edge_feature = edge_feature.repeat(1, self.embedding_dim, 1, 1).cuda()
                    
                    # Reshape cam_labels for WFF processing
                    cam_labels_2d = cam_labels.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    cam_labels_2d = F.interpolate(cam_labels_2d.float(), size=(28, 28), mode='bilinear', align_corners=False)
                    cam_labels_2d = cam_labels_2d.repeat(1, self.embedding_dim, 1, 1)  # [1, C, H, W]
                    
                    # Apply WFF fusion
                    enhanced_cam = self.wff(cam_labels_2d, edge_feature)
                    
                    # Convert back to original format
                    enhanced_cam = F.interpolate(enhanced_cam, size=(h, w), mode='bilinear', align_corners=False)
                    enhanced_cam = enhanced_cam.mean(dim=1).squeeze(0)  # [H, W]
                    cam_labels = enhanced_cam.argmax(dim=0)

            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)


        return seg, all_cam_labels, attn_pred


