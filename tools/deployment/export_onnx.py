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


def export_onnx_model(model, postprocessor, output_file, device="cpu"):
    """Export model and postprocessor to ONNX."""
    model = model.eval()
    if hasattr(model, "deploy"):
        model = model.deploy()
    if hasattr(postprocessor, "deploy"):
        postprocessor = postprocessor.deploy()

    class Model(nn.Module):
        def __init__(self, model, postprocessor) -> None:
            super().__init__()
            self.model = model
            self.postprocessor = postprocessor

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    export_model = Model(model, postprocessor).to(device)
    data = torch.rand(1, 3, 640, 640, device=device)
    size = torch.tensor([[640, 640]], device=device)
    _ = export_model(data, size)

    dynamic_axes = {"images": {0: "N"}, "orig_target_sizes": {0: "N"}}

    torch.onnx.export(
        export_model,
        (data, size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        do_constant_folding=True,
    )


def main(args):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)
    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print("not load model.state_dict, use default init state dict...")

    output_file = args.resume.replace(".pth", ".onnx") if args.resume else "model.onnx"

    export_onnx_model(cfg.model, cfg.postprocessor, output_file)

    if args.check:
        import onnx

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")

    if args.simplify:
        import onnx
        import onnxsim

        dynamic = True
        data = torch.rand(1, 3, 640, 640)
        size = torch.tensor([[640, 640]])
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
