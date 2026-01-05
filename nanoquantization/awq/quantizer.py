from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
from nanoquantization.awq.layers.gemm import WQLinear_GEMM
from nanoquantization.awq.utils.calibration_data import get_calibration_data
from nanoquantization.awq.utils.module import (
    get_named_linear,
    get_op_by_name,
    get_op_name,
    set_op_by_name,
)
from nanoquantization.layers.norm import RMSNorm
import torch.nn as nn
from transformers import AutoTokenizer
import torch


class AWQQuantizer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        w_bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        calibration_data: Union[str, List[str]] = "wikitext",
        hf_dataset_subset: str = "wikitext-103-v1",
        hf_dataset_split: str = "train",
        text_column: str = "text",
        n_samples: int = 128,
        max_seq_len: int = 512,
        apply_clip: bool = True,
        fake_quantization: bool = False,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.w_bits = w_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.calibration_data = calibration_data
        self.hf_dataset_subset = hf_dataset_subset
        self.hf_dataset_split = hf_dataset_split
        self.text_column = text_column
        self.n_samples = n_samples
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.apply_clip = apply_clip
        self.fake_quantization = fake_quantization
        self.output_dtype = output_dtype
        
    @torch.inference_mode()
    def _module_forward(
        self, module: nn.Module, x: torch.Tensor, module_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        return module(x, **module_kwargs)

    def _get_inputs(
        self, module: nn.Module, named_linears: Dict[str, nn.Linear]
    ) -> Dict[str, torch.Tensor]:
        """
        For a given module, we need to get the input tensor for the linear layers in the named_linears dictionary.
        """
        inputs = defaultdict(list)
        hooks = []

        # Register forward hooks on each linear layer to capture inputs
        def make_hook(name: str):
            def hook(linear_module, input, output):
                # the calibration data might be passed in mini-batches, so we accumate the inputs here and later concat them
                # input is a tuple of positional arguments passed to forward()
                if len(input) > 0 and input[0] is not None:
                    inputs[name].append(input[0].detach().clone())
                else:
                    raise ValueError(f"Hook for {name} received empty or None input")

            return hook

        for name, linear_layer in named_linears.items():
            hook = linear_layer.register_forward_hook(make_hook(name))
            hooks.append(hook)

        # Forward pass the module to trigger the hooks
        # update the inputs for the next transformer block
        self.inps = self._module_forward(
            module, self.inps, module_kwargs={"position_ids": self.position_ids}
        )

        # Remove all hooks
        for hook in hooks:
            hook.remove()

        def concat_and_assert(name: str, inputs: List[torch.Tensor]) -> torch.Tensor:
            inputs = torch.cat(inputs, dim=0)
            assert (
                inputs.shape[0] != 0
            ), f"input for {name} is empty, this might happen for MoE model and some experts are not used during calibration. Try increasing the calibration dataset size."
            return inputs

        return {name: concat_and_assert(name, inps) for name, inps in inputs.items()}

    @torch.inference_mode()
    def quantize(self):
        data = get_calibration_data(
            data=self.calibration_data,
            tokenizer=self.tokenizer,
            n_samples=self.n_samples,
            max_seq_len=self.max_seq_len,
            hf_dataset_subset=self.hf_dataset_subset,
            hf_dataset_split=self.hf_dataset_split,
            text_column=self.text_column,
        ).cuda(non_blocking=True)
        # the input for the first transformer block
        embed_output = self.model.get_embed_output(data)
        # shape: [1, sequence_length]
        self.position_ids = (
            torch.arange(data.shape[1], dtype=torch.int32)
            .unsqueeze(0)
            .cuda(non_blocking=True)
        )
        model_layers = self.model.get_model_layers()
        self.inps = embed_output
        for transformer_block in tqdm(model_layers, desc="Quantizing model layers"):
            named_linears = get_named_linear(transformer_block)
            inputs = self._get_inputs(transformer_block, named_linears)
            scaling_layers_config = self.model.get_layers_for_scaling(
                transformer_block,
                inputs,
                module_kwargs={"position_ids": self.position_ids},
            )
            # find the best weight scale factors for each transformer block
            scales = [
                self._find_best_scales(transformer_block, **config)
                for config in scaling_layers_config
            ]
            # apply the weight scale factors
            self._apply_scales(transformer_block, scales, inputs)
            # whether to clip max value for weight after applying the weight scale factors
            # the goal is to reduce the impact of extreme values in the weight when applying quantization later
            if self.apply_clip:
                # NOTE: some linear layers might not be applied weight scale factor
                clip_list = self._find_best_clip(named_linears, inputs)
                self._apply_clips(transformer_block, clip_list)

            if not self.fake_quantization:
                # apply the quantization for each linear layer
                self._apply_quant(transformer_block, named_linears)

    def _apply_quant(
        self, transformer_block: nn.Module, named_linears: Dict[str, nn.Linear]
    ):
        for name, linear_layer in named_linears.items():
            linear_layer.cuda()
            w = linear_layer.weight.data
            w, scales, zeros = self.pseudo_quantize_tensor(w)
            scales = scales.t().contiguous()
            zeros = zeros.t().contiguous()
            linear_layer.weight.data = w
            awq_linear = WQLinear_GEMM.from_linear(linear_layer, self.w_bits, self.group_size, scales, zeros, self.output_dtype)
            linear_layer.cpu()
            awq_linear.to(next(transformer_block.parameters()).device)
            set_op_by_name(transformer_block, name, awq_linear)
            

    @torch.inference_mode()
    def _apply_clips(
        self,
        module: nn.Module,
        clip_list: List[Tuple[str, torch.Tensor]],
    ):
        for clip_name, clip_tensor in clip_list:
            layer = get_op_by_name(module, clip_name)
            layer.to("cuda")
            clip_tensor.to("cuda")
            w = layer.weight.data
            # w is of shape [n_output, n_groups, group_size]
            w = w.reshape(w.shape[0], -1, self.group_size)
            # clip_tensor of shape [n_output, n_groups, 1]
            w = w.clamp(min=-clip_tensor, max=clip_tensor)
            w = w.reshape(w.shape[0], -1)
            layer.weight.data = w
            layer.cpu()
            clip_tensor.cpu()

    @torch.inference_mode()
    def _find_best_clip(
        self,
        named_linears: Dict[str, nn.Linear],
        inputs: Dict[str, torch.Tensor],
    ) -> List[Tuple[str, torch.Tensor]]:
        clip_list = []
        no_clip_list = [
            "q_",
            "k_",
        ]  # avoid clipping for query and key projection layers due to qk matmul
        for name, linear_layer in named_linears.items():
            if any(clip_name in name for clip_name in no_clip_list):
                continue
            inp = inputs[name]
            linear_layer.to("cuda")
            inp.to("cuda")
            # linear_layer.weight is of shape [n_output, n_input]
            # max_val is of shape [n_output, n_groups, 1] where n_input = n_groups * group_size
            max_val = self._compute_best_clip(linear_layer.weight, inp)
            clip_list.append((name, max_val))
            linear_layer.cpu()
            inp.cpu()
        return clip_list

    def _compute_best_clip(
        self,
        w: torch.Tensor,
        inp: torch.Tensor,
        n_grid: int = 20,
        max_shrink: float = 0.5,
        n_sample_tokens: int = 512,
    ) -> torch.Tensor:
        """
        Compute the max value clip for weight w for each group

        w: [n_output, n_input]
        inp: [batch_size, sequence_length, n_input]
        n_grid: number of grid points to search for the best clip
        max_shrink: the clip will never be smaller than max_shrink * the max value of w for each group
        n_sample_tokens: number of sample input to compute the best clip
        """
        assert w.ndim == 2, "weight should be 2D"
        w_orig_shape = w.shape

        if self.group_size <= 0:
            group_size = w.shape[-1]
        else:
            group_size = self.group_size
        assert (
            w.shape[-1] % group_size == 0
        ), "w.shape[-1] must be divisible by group_size"
        # First reshape the w and inp to for search best clip for each group
        w = w.reshape(
            w_orig_shape[0], 1, -1, self.group_size
        )  # [n_output, 1, n_groups, group_size]
        inp = inp.view(-1, inp.shape[-1])
        total_tokens = inp.shape[0]
        inp = inp.reshape(
            1, total_tokens, -1, group_size
        )  # [1, total_tokens, n_groups, group_size]
        # downsample the input to speed up the clipping
        step_size = max(1, total_tokens // n_sample_tokens)
        inp = inp[:, ::step_size]  # [1, n_sample_tokens, n_groups, group_size]

        orig_max_val = w.abs().amax(dim=-1, keepdim=True)  # [n_output, 1, n_groups, 1]
        orig_w = w.clone()  # [n_output, 1, n_groups, group_size]
        min_err = torch.ones_like(orig_max_val) * float("inf")
        best_clip = orig_max_val.clone()
        orig_out = (inp * orig_w).sum(dim=-1)  # [n_output, total_tokens, n_groups]
        for i in range(int(n_grid * max_shrink)):
            cur_max_val = orig_max_val * (1 - i / n_grid)  # [n_output, 1, n_groups, 1]
            cur_min_val = -cur_max_val  # [n_output, 1, n_groups, 1]
            w = torch.clamp(
                orig_w, min=cur_min_val, max=cur_max_val
            )  # [n_output, 1, n_groups, group_size]
            quantized_w = self.pseudo_quantize_tensor(w)[
                0
            ]  # [n_output, 1, n_groups, group_size]
            cur_out = (inp * quantized_w).sum(
                dim=-1
            )  # [n_output, n_sample_tokens, n_groups]
            cur_err = (
                (orig_out - cur_out).pow(2).mean(dim=1).view(min_err.shape)
            )  # [n_output, 1, n_groups, 1]
            ind = cur_err < min_err
            min_err[ind] = cur_err[ind]
            best_clip[ind] = cur_max_val[ind]
        return best_clip.squeeze(1)  # [n_output, n_groups, 1]

    def _scale_fc_fcs(
        self,
        prev_op: nn.Linear,
        quantized_layers: List[nn.Linear],
        weight_scales: torch.Tensor,
    ):
        prev_op.to("cuda")
        for layer in quantized_layers:
            layer.to("cuda")
        weight_scales = weight_scales.to(prev_op.weight.device)
        prev_op.weight[-weight_scales.shape[0] :].div_(weight_scales.view(-1, 1))
        if prev_op.bias is not None:
            prev_op.bias[-weight_scales.shape[0] :].div_(weight_scales.view(-1))

        for layer in quantized_layers:
            layer.weight.mul_(weight_scales.view(1, -1))

        for name, p in prev_op.named_parameters():
            assert (
                torch.isnan(p).sum() == 0
            ), f"prev_op {name} parameters should not contain nan"
        for layer in quantized_layers:
            for name, p in layer.named_parameters():
                assert (
                    torch.isnan(p).sum() == 0
                ), f"quantized layer {name} parameters should not contain nan"
        prev_op.cpu()
        for layer in quantized_layers:
            layer.cpu()
        weight_scales.cpu()

    def _scale_rms_fcs(
        self,
        prev_op: RMSNorm,
        quantized_layers: List[nn.Linear],
        weight_scales: torch.Tensor,
    ):
        prev_op.to("cuda")
        for layer in quantized_layers:
            layer.to("cuda")
        weight_scales = weight_scales.to(prev_op.weight.device)
        prev_op.weight.div_(weight_scales.view(-1))
        for layer in quantized_layers:
            layer.weight.mul_(weight_scales.view(1, -1))

        for name, p in prev_op.named_parameters():
            assert (
                torch.isnan(p).sum() == 0
            ), f"prev_op {name} parameters should not contain nan"
        for layer in quantized_layers:
            for name, p in layer.named_parameters():
                assert (
                    torch.isnan(p).sum() == 0
                ), f"quantized layer {name} parameters should not contain nan"
        prev_op.cpu()
        for layer in quantized_layers:
            layer.cpu()
        weight_scales.cpu()

    def _apply_scales(
        self,
        transformer_block: nn.Module,
        scales: List[Dict[str, Any]],
        inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        for item in scales:
            prev_op = get_op_by_name(transformer_block, item["prev_op"])
            quantized_layers = [
                get_op_by_name(transformer_block, layer)
                for layer in item["quantized_layers"]
            ]
            weight_scales = item["weight_scales"]

            if isinstance(prev_op, nn.Linear):
                self._scale_fc_fcs(prev_op, quantized_layers, weight_scales)
            elif isinstance(prev_op, RMSNorm):
                self._scale_rms_fcs(prev_op, quantized_layers, weight_scales)

            # divide the input by scales for the quantized layers to prepare for clipping
            if inputs is not None:
                for name in item["quantized_layers"]:
                    if name in inputs:
                        inp = inputs[name]
                        inp.div_(weight_scales.view(1, -1).to(inp.device))

    def _get_activation_scale(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.view(-1, inp.shape[-1]).abs().mean(dim=0)

    @torch.inference_mode()
    def _find_best_scales(
        self,
        transformer_block: nn.Module,
        prev_op: nn.Module,
        layers: List[nn.Module],
        inp: torch.Tensor,
        module2inspect: nn.Module = None,
        module_kwargs: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        if module2inspect is None:
            assert len(layers) == 1, "module2inspect is required for multiple layers"
            module2inspect = layers[0]
        activation_scale = self._get_activation_scale(inp)

        n_grid = 20  # search 20 values between 0 and 1
        original_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}
        # output tensor without any quantization
        orig_output = self._module_forward(module2inspect, inp, module_kwargs)
        history = []
        best_loss = float("inf")
        best_scales = None
        best_ratio = -1
        device = next(module2inspect.parameters()).device

        for i in range(n_grid):
            ratio = i / n_grid
            scales = activation_scale.pow(ratio).clamp(min=1e-4).view(-1).to(device)
            # normalize the scales
            scales = scales / (scales.max() * scales.min()).sqrt()
            for layer in layers:
                layer.weight.mul_(scales.view(1, -1))  # scale the weight first
                # simulate the quantization and de-quantization
                layer.weight.data = self.pseudo_quantize_tensor(layer.weight.data)[
                    0
                ] / scales.view(1, -1)
            quantized_output = self._module_forward(module2inspect, inp, module_kwargs)
            loss = (orig_output - quantized_output).float().pow(2).mean().item()
            history.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_scales = scales
                best_ratio = ratio
            # restore the original weights
            module2inspect.load_state_dict(original_sd)

        if best_ratio == -1:
            print(f"the loss history when searching for the best scales is: {history}")
            raise ValueError("Failed to find the best scales")
        assert (
            torch.isnan(best_scales).sum() == 0
        ), f"best_scales should not contain nan, but got {best_scales}"

        return dict(
            prev_op=get_op_name(transformer_block, prev_op),
            quantized_layers=[
                get_op_name(transformer_block, layer) for layer in layers
            ],
            weight_scales=best_scales.detach().cpu(),
        )

    def pseudo_quantize_tensor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate the quantization and de-quantization of a tensor to w_bits bits.
        """
        orig_shape = x.shape
        if self.group_size > 0:
            assert (
                x.shape[-1] % self.group_size == 0
            ), "x.shape[-1] must be divisible by group_size"
            x = x.view(-1, self.group_size)
        assert torch.isnan(x).sum() == 0, "weight should not contain nan"
        device = x.device

        if self.zero_point:
            # asymmetric quantization formula: q = round(x / scale) + zero_point
            # get the max for each group(row)
            max_val = x.amax(dim=-1, keepdim=True).to(device)
            # get the min for each group(row)
            min_val = x.amin(dim=-1, keepdim=True).to(device)
            max_quant_int = 2**self.w_bits - 1
            min_quant_int = 0
            # map the values between min_val and max_val to min_quant_int and max_quant_int
            scales = (max_val - min_val).clamp(min=1e-5) / max_quant_int
            # make sure the min_val is mapped to 0 which is the minimum quantized value
            zeros = (
                (-torch.round(min_val / scales))
                .clamp(min=min_quant_int, max=max_quant_int)
                .to(device)
            )
            # simulation quantization and de-quantization
            x = (torch.round(x / scales) + zeros).clamp(
                min=min_quant_int, max=max_quant_int
            )  # quantization
            x = (x - zeros) * scales  # de-quantization
            zeros = zeros.view(orig_shape[0], -1)
        else:
            # symmetric quantization formula: q = round(x / scale)
            max_quant_int = 2 ** (self.w_bits - 1) - 1
            min_quant_int = -(2 ** (self.w_bits - 1))
            max_val = x.abs().amax(dim=-1, keepdim=True).to(device)
            scales = max_val / max_quant_int
            # simulate quantization and de-quantization
            x = torch.round(x / scales).clamp(
                min=min_quant_int, max=max_quant_int
            )  # quantization
            x = x * scales  # de-quantization
            zeros = None

        x = x.reshape(orig_shape)
        scales = scales.view(orig_shape[0], -1)
        # NOTE: this scales is the block quantization scale, not the weight scale factors from activation magnitude
        return x, scales, zeros
