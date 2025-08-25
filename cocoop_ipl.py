import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

_tokenizer = _Tokenizer()

def read_word():
    df = pd.read_csv("filtered_candidates.csv")
    words = [str(w).lower() for w in df["word"].tolist() if isinstance(w, str)]
    return list(set(words))

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoCoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.classname = [c.replace("_", " ") for c in classnames]
        self.max_ctx = cfg.TRAINER.COCOOP.NUM_SEL_STEPS
        block_size = cfg.TRAINER.COCOOP.BLOCK_SIZE
        self.candidate_words = read_word()
        self.n_words = 1

        self.selected_indices = []
        self.selected_ctx = [f"a photo of a {name}, with emphasis on: " for name in self.classname]


        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
            
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.use_tk = cfg.TRAINER.IPL.USE_TOKEN
        self.atp_num = cfg.TRAINER.IPL.ATT_NUM

        print(f'self.use_atp is {self.use_tk}')

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        # if cfg.TRAINER.COCOOP.PREC == "fp16":
        #     self.meta_net.half()
        
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        if self.use_atp:

            # two attributes
            n_att1 = cfg.TRAINER.IPL.N_ATT1
            att1_text = '0'
            n_att2 = cfg.TRAINER.IPL.N_ATT2
            att2_text = '0'
            n_att3 = cfg.TRAINER.IPL.N_ATT3
            att3_text = '0'
            
            att_vectors_1 = torch.empty(n_att1, ctx_dim, dtype=dtype)
            att_vectors_2 = torch.empty(n_att2, ctx_dim, dtype=dtype)
            att_vectors_3 = torch.empty(n_att3, ctx_dim, dtype=dtype)
                            
            nn.init.normal_(att_vectors_1, std=0.01)
            prefix1 = " ".join(["X"] * n_att1)
            nn.init.normal_(att_vectors_2, std=0.01)
            prefix2 = " ".join(["X"] * n_att2)
            nn.init.normal_(att_vectors_3, std=0.01)
            prefix3 = " ".join(["X"] * n_att3)

            self.ctx_att1 = nn.Parameter(att_vectors_1)
            self.ctx_att2 = nn.Parameter(att_vectors_2) 
            self.ctx_att3 = nn.Parameter(att_vectors_3)
            
            if self.atp_num == 1:
                prompts = [prefix1 + " " + att1_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            elif self.atp_num == 2:
                prompts = [prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            elif self.atp_num == 3:
                prompts = [prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prefix3 + " " + att3_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            else:
                print("wrong parameter.")
        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])

        if self.use_tk:
            if self.atp_num == 1:
                self.register_buffer("token_middle1", embedding[:, 1+n_att1 : 1+n_att1+1, :])
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_ctx :, :])

            elif self.atp_num == 2:
                self.register_buffer("token_middle1", embedding[:, 1+n_att1 : 1+n_att1+1, :])
                self.register_buffer("token_middle2", embedding[:, 1+n_att1+1+n_att2 : 1+n_att1+1+n_att2+1, :])
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_att2+1+n_ctx :, :])

            elif self.atp_num == 3:
                self.register_buffer("token_middle1", embedding[:, 1+n_att1 : 1+n_att1+1, :])
                self.register_buffer("token_middle2", embedding[:, 1+n_att1+1+n_att2 : 1+n_att1+1+n_att2+1, :])
                self.register_buffer("token_middle3", embedding[:, 1+n_att1+1+n_att2+1+n_att3 : 1+n_att1+1+n_att2+1+n_att3+1, :])
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_att2+1+n_att3+1+n_ctx:, :])
        else:
            self.register_buffer("token_suffix", embedding[:, 1+n_ctx:, :])
            
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, prefix, ctx, suffix, att_ctx1=None, att_ctx2=None, att_ctx3=None, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        if self.use_atp:
            if self.atp_num == 1:
                middle_attribute1 = self.token_middle1
                prompts = torch.cat(
                    [
                        prefix,
                        att_ctx1,
                        middle_attribute1,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 2:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                prompts = torch.cat(
                    [
                        prefix,
                        att_ctx1,
                        middle_attribute1,
                        att_ctx2,
                        middle_attribute2,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 3:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                middle_attribute3 = self.token_middle3
                prompts = torch.cat(
                    [
                        prefix,
                        att_ctx1,
                        middle_attribute1,
                        att_ctx2,
                        middle_attribute2,
                        att_ctx3,
                        middle_attribute3,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
        else:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim) 
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx = self.ctx
        if self.use_atp:
            ctx_att1 = self.ctx_att1
            ctx_att2 = self.ctx_att2
            ctx_att3 = self.ctx_att3
            
            ctx_att1 = ctx_att1.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_att2 = ctx_att2.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_att3 = ctx_att3.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        # Use instance-conditioned context tokens for all classes
        prompts = []

        for index, ctx_shifted_i in enumerate(ctx_shifted):
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            if self.use_atp:
                pts_i = self.construct_prompts(prefix, ctx_i, suffix, ctx_att1, ctx_att2, ctx_att3)
            else:
                pts_i = self.construct_prompts(prefix, ctx_i, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts

    def greedy_select_next_token_global(self, image_features, image_labels, clip_model, gamma=0.1):
        clip_model = clip_model.to(self.device)

        if self.n_words > self.max_ctx:
            return  # already full

        best_token = None
        best_score = float("inf")

        for j, word in enumerate(self.candidate_words):
            if j in self.selected_indices:
                continue
            temp = [ctx + f"{word}" for ctx in self.selected_ctx]
            tokenized = clip.tokenize(temp)

            with torch.no_grad():
                text_features = clip_model.encode_text(tokenized.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = clip_model.logit_scale.exp() * image_features @ text_features.t()

            ce_loss = F.cross_entropy(logits, image_labels)
            diversity_penalty = self.compute_diversity_penalty(word, clip_model)
            total_score = ce_loss.item() + gamma * diversity_penalty

            if total_score < best_score:
                best_score = total_score
                best_token = j
                print('total', best_score, "ce_loss", ce_loss.item(), "penalty", diversity_penalty)
                print(temp[0])

        selected = self.candidate_words[best_token]
        self.selected_indices.append(best_token)
        # 将 best_token 拼入所有类别 ctx
        for i in range(self.n_cls):
            self.selected_ctx[i] = self.selected_ctx[i] + selected + ", "

        tok = clip.tokenize(selected).to(self.device)
        with torch.no_grad():
            self.word_embedding = clip_model.token_embedding(tok).expand(self.n_cls, -1, -1)  # Get the embedding
        emd = self.word_embedding[:, 1:2, :]
        # Replace mid_texti with the selected word's embedding
        # Replace mid_text1, mid_text2, etc. for each class
        if self.n_words == 1:  # Replace for the first class, you can modify the conditions as needed
            self.token_middle1 = emd  # Replacing mid_text1 with the word's embedding
        elif self.n_words == 2:  # Similarly for other classes, adjust conditions accordingly
            self.token_middle2 = emd
        elif self.n_words == 3:
            self.token_middle3 = emd
        elif self.n_words == 4:
            self.token_middle4 = emd
        elif self.n_words == 5:
            self.token_middle5 = emd
        self.n_words += 1
        print(self.candidate_words[best_token])

        accuracy = self.compute_accuracy(image_features, image_labels, clip_model)
        print(accuracy)

    def compute_diversity_penalty(self, candidate_word, clip_model):
        """
        计算多样性惩罚：候选 token 与当前已选 prompt token 的相似度均值。
        相似度越高，惩罚越大。
        """
        token = clip.tokenize(candidate_word).to(self.device)
        with torch.no_grad():
            candidate_embed = clip_model.encode_text(token)  # [1, D]
            candidate_embed = candidate_embed / candidate_embed.norm(dim=-1, keepdim=True)

        selected_words = [self.candidate_words[idx] for idx in self.selected_indices]
        if not selected_words:
            return 0.0

        tokenized = clip.tokenize(selected_words).to(self.device)
        with torch.no_grad():
            selected_embeds = clip_model.encode_text(tokenized)  # [N, D]
            selected_embeds = selected_embeds / selected_embeds.norm(dim=-1, keepdim=True)

        sim = cosine_similarity(candidate_embed.cpu().numpy(), selected_embeds.cpu().numpy())
        penalty = sim.mean().item()
        return penalty

    def compute_accuracy(self, image_features, image_labels, clip_model):

        # 计算文本特征
        tokenized = clip.tokenize(self.selected_ctx).to(self.device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # 计算相似度
        logits = clip_model.logit_scale.exp() * image_features @ text_features.t()
        # 获取预测标签
        preds = logits.argmax(dim=-1)
        # 计算准确率
        accuracy = (preds == image_labels).float().mean().item()
        return accuracy


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class CoCoOp_ATP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        #if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
        clip_model.float()
        self.clip_model = clip_model.to(self.device)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def _extract_all_image_features(self):
        feats, labs = [], []
        self.model.eval()
        for batch in self.train_loader_x:
            x, y = self.parse_batch_train(batch)
            feats.append(self.model.image_encoder(x))
            labs.append(y)
        return torch.cat(feats, 0), torch.cat(labs, 0)

    def _save_selected_prompt(self, cfg):
        sd = osp.join(cfg.OUTPUT_DIR, "selected_prompt")
        os.makedirs(sd, exist_ok=True)
        torch.save({
            "ctx_text": getattr(self.model.prompt_learner, "selected_indices", None),
            "ctx": self.model.prompt_learner.ctx.detach().cpu(),
        }, osp.join(sd, "prompt.pth"))
        print(f"Saved to {sd}/prompt.pth")

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_middle1" in state_dict:
                del state_dict["token_middle1"]
            if "token_middle2" in state_dict:
                del state_dict["token_middle2"]
            if "token_middle3" in state_dict:
                del state_dict["token_middle3"]
            
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
