"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn

from src.core import YAMLConfig


def main(
    args,
):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print("not load model.state_dict, use default init state dict...")

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    data = torch.rand(32, 3, 640, 640)
    size = torch.tensor([[640, 640]])
    _ = model(data, size)

    dynamic_axes = {
        "images": {
            0: "N",
        },
        "orig_target_sizes": {0: "N"},
    }

    output_file = args.resume.replace(".pth", ".onnx") if args.resume else "model.onnx"

    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")

    if args.simplify:
        import onnx
        import onnxsim

        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {"images": data.shape, "orig_target_sizes": size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f"Simplify onnx model {check}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/dfine/dfine_hgnetv2_l_coco.yml",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()
    main(args)




import copy
def export_onnx_model(model: nn.Module,
                      postprocessor: nn.Module,
                      output_file: str,
                      device: str = "cpu",
                      opset: int = 16,
                      simplify: bool = True) -> bool:
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError:
        print("ONNX not installed; skipping export.")
        return False

    import copy
    model_c = copy.deepcopy(model).to(device).eval()
    post_c  = copy.deepcopy(postprocessor).to(device).eval()

    if hasattr(model_c, "deploy"):
        model_c = model_c.deploy()
    if hasattr(post_c, "deploy"):
        post_c = post_c.deploy()

    class ExportWrapper(nn.Module):
        def __init__(self, m, p):
            super().__init__()
            self.m = m
            self.p = p
        def forward(self, images, orig_target_sizes):
            out = self.m(images)
            return self.p(out, orig_target_sizes)

    wrapper = ExportWrapper(model_c, post_c).to(device)

    # ----- make dummy inputs CONSISTENT -----
    N = 8  # representative batch; can be 1 if you prefer
    images = torch.rand(N, 3, 640, 640, device=device, dtype=torch.float32)
    # IMPORTANT: int64 + same N
    sizes  = torch.full((N, 2), 640, device=device, dtype=torch.long)

    dynamic_axes = {"images": {0: "N"}, "orig_target_sizes": {0: "N"}}

    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (images, sizes), output_file,
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    if simplify:
        try:
            import onnx, onnxsim
            # Keep simplification dynamic-shape safe; don't pass fixed shapes.
            model_simplified, ok = onnxsim.simplify(
                output_file,
                dynamic_input_shape=True
            )
            onnx.save(model_simplified, output_file)
            print(f"Simplified ONNX: {ok}")
        except ModuleNotFoundError:
            print("onnxsim not installed; exported unsimplified ONNX.")
    return True