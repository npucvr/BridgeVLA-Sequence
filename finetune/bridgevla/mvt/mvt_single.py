'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/mvt/mvt_single.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
# gbw___
import os
#____
import torch
from torch import nn
from einops import rearrange
import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.attn import (
    FixedPositionalEncoding,
)
from bridgevla.mvt.raft_utils import ConvexUpSample
from PIL import Image
# gbw____
from bridgevla.mvt.filt3r_akf import A1UnifiedTokenKalmanFilter
# ____



class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        img_feat_dim,
        feat_dim,
        im_channels,
        activation,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        rend_three_views,
        use_point_renderer,
        pe_fix,
        feat_ver,
        wpt_img_aug,
        inp_pre_pro,
        inp_pre_con,
        cvx_up,
        xops,
        rot_ver,
        num_rot,
        renderer_device="cuda:0",
        renderer=None,
        no_feat=False,
        load_pretrain=False,
        pretrain_path=None,
    ):
        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix
        self.feat_ver = feat_ver
        self.wpt_img_aug = wpt_img_aug
        self.inp_pre_pro = inp_pre_pro
        self.inp_pre_con = inp_pre_con
        self.cvx_up = cvx_up
        self.use_point_renderer = use_point_renderer
        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.no_feat = no_feat

        if self.cvx_up:
            assert not self.inp_pre_con, (
                "When using the convex upsampling, we do not concatenate"
                " features from input_preprocess to the features used for"
                " prediction"
            )

        print(f"MVT Vars: {vars(self)}")

        assert not renderer is None
        self.renderer = renderer
        self.num_img = self.renderer.num_img
        # Modify it to adapt to vlm. 16**2 is the number of patches in the image
        self.num_pat_img = 16  

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1


        # Hardcoded for vlm
        self.vlm_dim=2048  

        # gbw____
        # up0 是训练好的 ConvexUpSample。A1 插入在它之前：先得到 PaliGemma
        # image tokens，再做 AKF，最后把滤波后的 token 送入 up0。
        # ____
        self.up0 = ConvexUpSample(
            in_dim=self.vlm_dim,
            out_dim=1,
            up_ratio=self.img_patch_size,
        )

        if not self.no_feat:
            feat_fc_dim = 0
            feat_fc_dim += self.vlm_dim
            # Because we will concatenate the max-pooled image tokens and the image tokens corresponding to the waypoint later.
            if self.cvx_up:
                feat_fc_dim += self.vlm_dim
            else:
                feat_fc_dim += self.final_dim
            

            def get_feat_fc(
                _feat_in_size,
                _feat_out_size,
                _feat_fc_dim=feat_fc_dim,
            ):
                """
                _feat_in_size: input feature size
                _feat_out_size: output feature size
                _feat_fc_dim: hidden feature size
                """
                layers = [
                    nn.Linear(_feat_in_size, _feat_fc_dim),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim, _feat_fc_dim // 2),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim // 2, _feat_out_size),
                ]
                feat_fc = nn.Sequential(*layers)
                return feat_fc

            feat_out_size = feat_dim

            if self.rot_ver == 0:
                self.feat_fc = get_feat_fc(
                    self.num_img * feat_fc_dim,
                    feat_out_size,
                )
            elif self.rot_ver == 1:
                assert self.num_rot * 3 <= feat_out_size
                feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3)
                if feat_out_size_ex_rot > 0:
                    self.feat_fc_ex_rot = get_feat_fc(
                        self.num_img * feat_fc_dim, feat_out_size_ex_rot
                    )

                self.feat_fc_init_bn = nn.BatchNorm1d(self.num_img * feat_fc_dim)
                self.feat_fc_pe = FixedPositionalEncoding(
                    self.num_img * feat_fc_dim, feat_scale_factor=1
                )
                self.feat_fc_x = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_y = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_z = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)

            else:
                assert False

        if self.use_point_renderer:
            from point_renderer.rvt_ops import select_feat_from_hm
        else:
            from bridgevla.mvt.renderer import select_feat_from_hm

        from transformers import (
            PaliGemmaProcessor,
            PaliGemmaForConditionalGeneration,
        )
        from safetensors import safe_open
        import json

        def load_all_params(checkpoint_dir):
            # Load the index file
            with open(f"{checkpoint_dir}/model.safetensors.index.json") as f:
                index = json.load(f)
            
            all_params = {}
            for shard_file in set(index["weight_map"].values()):
                with safe_open(f"{checkpoint_dir}/{shard_file}", framework="pt") as f:
                    for key in f.keys():
                        # Remove the "module." prefix
                        clean_key = key.replace("module.", "")
                        all_params[clean_key] = f.get_tensor(key)
            return all_params


        # gbw___
        model_id = os.environ.get(
            "BRIDGEVLA_PALIGEMMA_PATH", "google/paligemma-3b-pt-224"
        )
        local_files_only = os.path.isdir(model_id)
        model_load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "local_files_only": local_files_only,
        }
        #____
        if load_pretrain:
            assert pretrain_path is not None

            # gbw___
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, **model_load_kwargs
            )
            self.processor = PaliGemmaProcessor.from_pretrained(
                model_id, local_files_only=local_files_only
            )
            #____
            pretrained_dir=pretrain_path
            print("The pretrained path is:",pretrained_dir)
            all_params = load_all_params(pretrained_dir)

            # Separate the base model parameters (assuming the original model parameter names do not contain "up0")
            base_params = {k: v for k, v in all_params.items() if not k.startswith("up0.")}

            # Separate the custom layer parameters
            custom_params = {k.replace("up0.",""): v for k, v in all_params.items() if k.startswith("up0.")}
            # Load parameters (strict mode)
            missing_keys, unexpected_keys = self.model.load_state_dict(base_params, strict=False)
            print("Missing keys  base:", missing_keys)  # Should be an empty list
            print("Unexpected keys base:", unexpected_keys) # Should be an empty list
            # Load parameters
            missing_keys_up0, unexpected_keys_up0 = self.up0.load_state_dict(custom_params, strict=True)
            print("Missing keys up0:", missing_keys_up0)  # Should be an empty list
            print("Unexpected keys up0 :", unexpected_keys_up0) # Should be an empty list
            import time
            time.sleep(5)

            
        else:

            # gbw___
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, **model_load_kwargs
            )
            self.processor = PaliGemmaProcessor.from_pretrained(
                model_id, local_files_only=local_files_only
            )
            #____
            print("You are loading original paligemma model!")

        # gbw____
        # 每个 MVT 实例创建一个 A1 filter，在多个时间步之间保留 temporal state。
        # FILTER_MODE=none 时仍创建轻量对象，但 apply() 完全返回原 token，不改变 baseline。
        self._filt3r_token_filter = A1UnifiedTokenKalmanFilter()
        self._last_filt3r_diagnostics = None
        # gbw____
        self._last_filt3r_raw_shadow_waypoint = None
        # ____
        # ____
        global select_feat_from_hm

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        Transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    @staticmethod
    def trans_cuda_tensor_2_PIL(cuda_tensor):
        # Default c,h,w, and 0,1
        # 1. Move the tensor from GPU to CPU
        tensor_cpu = cuda_tensor.cpu()

        # 2. Convert to a numpy array and adjust the dimension order [3, 224, 224] -> [224, 224, 3]
        image = tensor_cpu.permute(1, 2, 0).numpy()

        # 3. Convert the values from [0, 1] to integers in [0, 255] and cast to uint8 type
        image = (image * 255).astype('uint8')

        # 4. Create a PIL image object
        pil_image = Image.fromarray(image)

        # 5. Convert to RGB format (ensure the image is RGB)
        pil_image_rgb = pil_image.convert("RGB")
        return pil_image_rgb

    def forward(
        self,
        img,
        wpt_local=None,
        rot_x_y=None,
        language_goal=None,
        forward_no_feat=False,
        # gbw____
        filter_stage=1,
        # ____
        **kwargs,
    ):
        """
        这个函数完成单个 MVT stage 的一次前向推理。

        需要先记住它的主线：

        ``输入图像特征 → RGB 图像 → PaliGemma → 最后一层 hidden state
        → 选择 image tokens → 可选 AKF → ConvexUpSample → heatmap``。

        参数 ``img`` 的形状是 ``(B, num_img, C, H, W)``。其中当前模型
        ``num_img=3``、``H=W=224``，输入通道中的 ``3:6`` 是 RGB；
        PaliGemma 最后一层 hidden state 的形状是 ``(B,L,2048)``。
        每个 view 有 ``16*16=256`` 个视觉 token，所以 3 个 view 一共选择
        ``768`` 个 image token，随后恢复成 ``(B,3,16,16,2048)``。

        ``forward_no_feat=True`` 时只计算 heatmap，供 Stage-1 先预测 waypoint；
        ``forward_no_feat=False`` 时还会根据 waypoint 从 token feature 中取特征，
        供旋转等离散动作头使用。``filter_stage`` 只决定当前调用是否经过同一个
        temporal AKF，默认 Stage-1 过滤，Stage-2 保持 raw。
        """

        # gbw____
        # 1. 读取并检查输入形状。
        # num_img、img_size 必须与模型初始化时的 renderer 配置一致，否则后面的
        # PaliGemma image-token 数量和 ConvexUpSample reshape 都会错位。
        # ____
        bs, num_img, img_feat_dim, h, w = img.shape
        assert num_img == self.num_img
        assert h == w == self.img_size
        # gbw____
        self._last_filt3r_raw_shadow_waypoint = None
        # ____
        # gbw____
        temporal_step = kwargs.pop("temporal_step", None)
        task_label = None
        if language_goal is not None:
            try:
                task_label = str(language_goal[0][0])
            except (IndexError, KeyError, TypeError):
                task_label = str(language_goal)
        self._filt3r_token_filter.set_context(
            task_label=task_label, timestep=temporal_step
        )
        # ____
        # gbw____
        # 这里的 task_label/timestep 只写入 diagnostics，不参与滤波参数选择。
        # temporal_step 由 agent.act() 传入，用于标记当前 observation 在 episode
        # 中的位置；真正的 state 生命周期仍由 reset_temporal_state() 控制。
        # ____
        # only use rgb part
        # ____
        # gbw____
        # 2. 将 MVT 的多视角输入转换为 PaliGemma 需要的格式。
        # img 原本可能包含 RGB、深度或其他 renderer 特征；这里仅保留通道 3:6
        # 的 RGB，得到 (B,3,3,224,224)，即每个样本 3 个 view、每个 view 3 个颜色通道。
        # PaliGemmaProcessor 接收 PIL 图像和语言 prompt，因此每个 RGB tensor 会
        # 转成一张 PIL 图像；这个转换只服务于 VLM 输入，不改变后面的 MVT tensor。
        # ____
        img = img[:,:, 3:6, :, :] # bs,3,3,224,224

        prompts =[ text[0][0] for text in language_goal]
        images = [[MVT.trans_cuda_tensor_2_PIL(example)for example in examples] for examples in img]

        assert len(prompts)==len(images)
        # gbw____
        # processor 会同时构造 input_ids、attention_mask 和视觉输入。padding 只
        # 影响序列长度 L，不应该影响视觉 token 的选择，所以后面会使用 image_token_id
        # 和 attention_mask 做显式索引，而不是简单截取 hidden state 的前缀。
        # ____
        model_inputs = self.processor(text=prompts, images=images, return_tensors="pt",padding="longest")
        model_inputs = model_inputs.to(self.model.dtype).to(self.model.device)
        # gbw____
        # PaliGemma 推理。output_hidden_states=True 是关键：A1/B1/B2/B3 需要最后一层
        # hidden state，而不是只使用模型的语言 logits。这里的输出仍包含文本 token
        # 和 image token，尚未完成视觉 token 的筛选。
        # ____
        outputs = self.model(**model_inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  

        # gbw____
        # 3. 取 PaliGemma 最后一层 hidden state：x.shape=(B,L,2048)。
        # L 由语言 prompt、padding 和视觉 token 共同决定；A1 不使用文本 token，
        # 只使用后面按照 image_token_id 找出的视觉 token。
        # ____
        x = hidden_states[-1]  # get the features of the last layer


        # gbw____
        # 4. 找到 PaliGemma 的 image_token_id。优先从 processor 读取，若 processor
        # 没有暴露该字段，再从 model.config 读取；两处都没有时继续运行会有 token
        # 错位风险，因此直接报错。
        # ____
        image_tokens= []
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError("PaliGemma image_token_id is unavailable")
        # ____

        # gbw____
        # 5. 对 batch 中每个样本单独选择视觉 token。
        # current_ids 给出每个序列位置的 token 类型，current_attention 排除 padding；
        # current_output 是该样本的 hidden state，形状为 (L,2048)。
        # ____
        for i in range(bs):
            # Get the ids and output of the current batch
            current_ids = model_inputs["input_ids"][i]
            current_attention = model_inputs["attention_mask"][i]
            current_output = x[i]

            # 只选 active 的 image token。不能用 x[i, :768] 替代，因为文本 prompt
            # 或 padding 的长度可能改变视觉 token 在序列中的位置。
            image_indices = torch.nonzero(
                (current_ids == image_token_id) & (current_attention != 0),
                as_tuple=True,
            )[0]
            expected_image_tokens = 256 * self.num_img
            # gbw____
            # 当前模型有 3 个 view，每个 view 是 16x16=256 个视觉 token，
            # 所以每个样本必须选出 768 个 token；不够或多余时直接报错，避免错位滤波。
            # 每个 token 的 hidden width 仍然是 2048。
            # ____
            if image_indices.numel() != expected_image_tokens:
                raise RuntimeError(
                    "PaliGemma image-token count mismatch: "
                    f"expected {expected_image_tokens}, got {image_indices.numel()}"
                )
            non_zero_output = current_output[image_indices]

            # non_zero_output.shape=(768,2048)，保留当前样本的全部视觉 token。
            image_tokens.append(non_zero_output)

        # gbw____
        self._filt3r_token_filter.record_token_selection_diagnostics(
            image_token_id=int(image_token_id),
            selected_count=int(image_tokens[0].shape[0]),
            expected_count=int(expected_image_tokens),
            sequence_length=int(model_inputs["input_ids"].shape[1]),
            stage=filter_stage,
        )
        # ____

        # gbw____
        # 6. 拼回 batch 维度：image_tokens.shape=(B,768,2048)。到这里仍然是
        # PaliGemma 的 raw image token，还没有经过 Kalman filter。
        # ____
        image_tokens = torch.stack(image_tokens)
        # gbw____
        # 7. 选择 A0 旁路或 A1/B1/B2/B3 滤波路径。
        # 训练阶段不允许启用 temporal filter，因为 state 是推理时按时间步维护的，
        # 不能把跨样本、跨 batch 的历史状态带入训练反向传播。
        # ____
        if self._filt3r_token_filter.enabled and self.training:
            raise RuntimeError(
                "FILTER_MODE=filt3r_akf is inference-only for A1"
            )
        if self._filt3r_token_filter.enabled_for_stage(filter_stage):
            # gbw____
            # 过滤路径：先把序列布局恢复成 canonical Z_t=(B,3,16,16,2048)，
            # 再调用同一个 filter。返回值仍是相同 shape 的 filtered token S_t。
            # 这正是整个迁移的插入点：PaliGemma image tokens → AKF → ConvexUpSample。
            # ____
            # raw_canonical_tokens 是当前帧 measurement Z_t。
            # ____
            raw_canonical_tokens = rearrange(
                image_tokens,
                "b (c h1 h2) d -> b c h1 h2 d",
                c=self.num_img,
                h1=self.num_pat_img,
                h2=self.num_pat_img,
            )
            canonical_tokens = raw_canonical_tokens
            canonical_tokens = self._filt3r_token_filter.apply(
                canonical_tokens, stage=filter_stage
            )
            self._last_filt3r_diagnostics = (
                self._filt3r_token_filter.last_diagnostics
            )
            x = canonical_tokens.permute(0, 4, 1, 2, 3)
        else:
            # gbw____
            # A0 或未启用的 stage：不创建/读取 temporal state，沿用官方 raw token
            # 的 reshape。这里必须保持 exact bypass，确保 A0 与原始 baseline 一致。
            # ____
            self._last_filt3r_diagnostics = None
            x = rearrange(
                image_tokens,
                "b (c h1 h2) w -> b w c h1 h2",
                c=self.num_img,
                h1=self.num_pat_img,
                h2=self.num_pat_img,
            )
        # ____
        # gbw____
        # 8. 在送入 up0 前，把 token layout 转成 ConvexUpSample 需要的图像特征格式。
        # filter 输出是 (B,2048,3,16,16)，transpose 后按 view 合并为
        # x.shape=(B*3,2048,16,16)。每个 view 作为一个独立的 2048-channel feature map。
        # _feat 是对空间网格做 max pooling 得到的全局 feature，只有在
        # forward_no_feat=False 时才会继续与 waypoint feature 拼接。
        # ____
        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.vlm_dim, self.num_pat_img, self.num_pat_img
            )
        )
        x=x.to(torch.float32)

        # gbw____
        # 9. 唯一一次 ConvexUpSample 调用。
        # 输入是 (B*3,2048,16,16)，up_ratio=16 后得到每个 view 的低/高分辨率
        # heatmap logits，随后恢复为 trans.shape=(B,3,224,224)。A1/B1/B2/B3
        # 只改变进入这里的 token，不改变 up0 的参数或 heatmap decoder 结构。
        # ____
        trans = self.up0(x)
        trans = trans.view(bs, self.num_img, h, w)
        # gbw____
        # 10. 可选 raw shadow 只用于 diagnostics。
        # 当 FILTER_DIAGNOSTICS_SHADOW_RAW=1 时，raw_canonical_tokens 会额外走一次
        # 同一个 up0，用来比较 raw/filtered heatmap peak 和 waypoint；它不写入 AKF
        # state，也不参与最终 action。正常运行只使用上面的 filtered trans。
        # ____
        # gbw____
        raw_shadow_trans = None
        if (
            self._filt3r_token_filter.shadow_raw_enabled
            and self._filt3r_token_filter.enabled_for_stage(filter_stage)
        ):
            raw_x = raw_canonical_tokens.permute(0, 4, 1, 2, 3)
            raw_x = (
                raw_x.transpose(1, 2)
                .clone()
                .view(
                    bs * self.num_img,
                    self.vlm_dim,
                    self.num_pat_img,
                    self.num_pat_img,
                )
                .to(torch.float32)
            )
            raw_shadow_trans = self.up0(raw_x).view(bs, self.num_img, h, w)
            self._last_filt3r_raw_shadow_waypoint = self.get_wpt(
                out={"trans": raw_shadow_trans.clone().detach()},
                dyn_cam_info=None,
            )
        # ____
        # gbw____
        # 记录 filtered heatmap 的 peak、margin、entropy；如果启用了 shadow，同时记录
        # raw 与 filtered peak 的位移。diagnostics 是旁路文件，不会改变 out。
        self._filt3r_token_filter.record_output_diagnostics(
            trans,
            stage=filter_stage,
            raw_logits=raw_shadow_trans,
        )
        # ____


        # gbw____
        # 11. 根据 forward_no_feat 决定是否继续生成训练用的辅助 feature。
        # Stage-1 在 Stage-2 模式下通常以 forward_no_feat=True 调用：此时只需要
        # trans 来预测 waypoint；完整动作头调用时才进入下面的 feature 分支。
        # ____
        if not forward_no_feat:

            # 评估时没有 ground-truth wpt_local，需要先从当前 heatmap 得到预测 waypoint。
            # 训练时 wpt_local 由数据集提供，因此保留传入的 ground truth，不从 heatmap
            # 反推训练标签。
            if not self.training:
                wpt_local = self.get_wpt(
                    out={"trans": trans.clone().detach()},
                    dyn_cam_info=None,
                )

            # 12. 将 3D waypoint 投影回三个 camera 的 2D 图像坐标，得到
            # (B,1,3,2)，再 reshape 为 (B*3,2)。后面用这个位置从低分辨率 token
            # feature map 中取出 waypoint 对应的局部 feature。
            wpt_img = self.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                dyn_cam_info=None,
            )
            wpt_img = wpt_img.reshape(bs * self.num_img, 2)

            # 训练时对 waypoint image 坐标加随机扰动，模拟定位误差；评估时不加噪声。
            if self.training:
                wpt_img = mvt_utils.add_uni_noi(
                    wpt_img, self.wpt_img_aug * self.img_size
                )
                wpt_img = torch.clamp(wpt_img, 0, self.img_size - 1)

            _wpt_img = wpt_img / self.img_patch_size
            _u = x
            assert (
                0 <= _wpt_img.min() and _wpt_img.max() <= x.shape[-1]
            ), print(_wpt_img, x.shape)

            _wpt_img = _wpt_img.unsqueeze(1)
            _feat = select_feat_from_hm(_wpt_img, _u)[0]
            _feat = _feat.view(bs, -1)
            feat.append(_feat)
            feat = torch.cat(feat, dim=-1)

            # gbw____
            # 13. 旋转头有两种实现：rot_ver=0 使用一个联合 feature_fc；rot_ver=1
            # 使用自回归式 x→y→z 离散旋转预测。这里的 feat 已经包含全局 token feature
            # 和 waypoint 对应的局部 feature，最终 out 保存给上层 get_q() 使用。
            # ____
            if self.rot_ver == 0:
                feat = self.feat_fc(feat)
                out = {"feat": feat}
            elif self.rot_ver == 1:
                # features except rotation
                feat_ex_rot = self.feat_fc_ex_rot(feat)

                # batch normalized features for rotation
                feat_rot = self.feat_fc_init_bn(feat)
                # feat_rot = self.feat_fc_init_bn(feat)
                feat_x = self.feat_fc_x(feat_rot)

                if self.training:
                    rot_x = rot_x_y[..., 0].view(bs, 1)
                else:
                    # sample with argmax
                    rot_x = feat_x.argmax(dim=1, keepdim=True)

                # rot_x_pe = self.feat_fc_pe(rot_x).to(torch.bfloat16)
                rot_x_pe = self.feat_fc_pe(rot_x)
                feat_y = self.feat_fc_y(feat_rot + rot_x_pe)

                if self.training:
                    rot_y = rot_x_y[..., 1].view(bs, 1)
                else:
                    rot_y = feat_y.argmax(dim=1, keepdim=True)
                rot_y_pe = self.feat_fc_pe(rot_y)
                # rot_y_pe = self.feat_fc_pe(rot_y).to(torch.bfloat16)
                feat_z = self.feat_fc_z(feat_rot + rot_x_pe + rot_y_pe)
                out = {
                    "feat_ex_rot": feat_ex_rot,
                    "feat_x": feat_x,
                    "feat_y": feat_y,
                    "feat_z": feat_z,
                }
        
        else:
            # 只请求 heatmap 时不创建辅助 feature，保持输出最小化。
            out = {}

        # gbw____
        # 14. 无论是否计算辅助 feature，trans 都必须返回给上层。上层会用 trans
        # 做 heatmap softmax、3D waypoint 解码以及后续 rotation/gripper/collision
        # action 预测。
        # ____
        out.update({"trans": trans})

        return out





    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        if self.use_point_renderer:
            pred_wpt = pred_wpt.squeeze(1)

        assert y_q is None

        return pred_wpt

    # gbw____
    def get_filt3r_diagnostics(self):
        return self._last_filt3r_diagnostics

    # gbw____
    def get_filt3r_raw_shadow_waypoint(self):
        return self._last_filt3r_raw_shadow_waypoint
    # ____

    def reset_temporal_state(self, reason="agent_reset"):
        # gbw____
        # 上层 agent/evaluator 在 episode、task switch、terminal 或 timeout 时调用，
        # 将 A1 的历史 token state 清空。
        # ____
        return self._filt3r_token_filter.reset(reason=reason)

    # gbw____
    def record_waypoint_diagnostics(
        self, waypoint, stage=1, raw_waypoint=None
    ):
        return self._filt3r_token_filter.record_waypoint_diagnostics(
            waypoint, stage=stage, raw_waypoint=raw_waypoint
        )
    # ____

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")


        self.renderer.free_mem()
