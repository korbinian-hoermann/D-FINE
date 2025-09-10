"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time

import torch
import torch.nn as nn

from ..misc import dist_utils, stats
from ..data import CocoEvaluator, get_coco_api_from_dataset
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch, evaluate_onnx


class DetSolver(BaseSolver):
    def _export_to_onnx(self, module: nn.Module, output_file):
        model = module.eval()
        if hasattr(model, "deploy"):
            model = model.deploy()
        postprocessor = self.postprocessor
        if hasattr(postprocessor, "deploy"):
            postprocessor = postprocessor.deploy()

        class Model(nn.Module):
            def __init__(self, model, postprocessor):
                super().__init__()
                self.model = model
                self.postprocessor = postprocessor

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                return self.postprocessor(outputs, orig_target_sizes)

        export_model = Model(model, postprocessor).to(self.device)
        data = torch.rand(1, 3, 640, 640, device=self.device)
        size = torch.tensor([[640, 640]], device=self.device)
        _ = export_model(data, size)
        dynamic_axes = {"images": {0: "N"}, "orig_target_sizes": {0: "N"}}
        torch.onnx.export(
            export_model,
            (data, size),
            str(output_file),
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes=dynamic_axes,
            opset_version=16,
            do_constant_folding=True,
        )

    def _save_and_log_best(self, module: nn.Module, ckpt_path):
        dist_utils.save_on_master(self.state_dict(), ckpt_path)
        if self.use_mlflow and dist_utils.is_main_process():
            self.mlflow.log_artifact(str(ckpt_path))

    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        if self.use_mlflow:
            self.mlflow.set_experiment(args.yaml_cfg.get("mlflow_experiment", "default"))
            self.mlflow.start_run(run_name=args.yaml_cfg.get("mlflow_run_name"))
            flat_params = {
                k: v
                for k, v in args.yaml_cfg.items()
                if isinstance(v, (int, float, str, bool))
            }
            if flat_params:
                self.mlflow.log_params(flat_params)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_mlflow
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                use_mlflow=self.use_mlflow,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_mlflow,
                output_dir=self.output_dir,
            )

            # TODO
            for k in test_stats:
                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            self._save_and_log_best(
                                module, self.output_dir / "best_stg2.pth"
                            )
                        else:
                            self._save_and_log_best(
                                module, self.output_dir / "best_stg1.pth"
                            )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            self._save_and_log_best(
                                module, self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        self._save_and_log_best(
                            module, self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        "epoch": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_mlflow:
                logs = {
                    f"val/{metric_names[idx]}": test_stats["coco_eval_bbox"][idx]
                    for idx in range(len(metric_names))
                }
                logs["epoch"] = epoch
                self.mlflow.log_metrics(logs, step=epoch)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        if getattr(self, "test_dataloader", None) is not None:
            best_ckpt = self.output_dir / "best_stg2.pth"
            if not best_ckpt.exists():
                best_ckpt = self.output_dir / "best_stg1.pth"
            if best_ckpt.exists():
                print(f"Evaluating {best_ckpt} on test dataset")
                self.load_resume_state(str(best_ckpt))
                module = self.ema.module if self.ema else self.model
                coco_gt = get_coco_api_from_dataset(self.test_dataloader.dataset)
                test_evaluator = CocoEvaluator(coco_gt=coco_gt, iou_types=self.evaluator.iou_types)
                test_stats, _ = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.test_dataloader,
                    test_evaluator,
                    self.device,
                    epoch=self.last_epoch,
                    use_mlflow=self.use_mlflow,
                )
                if self.use_mlflow and "coco_eval_bbox" in test_stats:
                    logs = {
                        f"test/{metric_names[idx]}": test_stats["coco_eval_bbox"][idx]
                        for idx in range(len(metric_names))
                    }
                    self.mlflow.log_metrics(logs, step=self.last_epoch)
                onnx_path = best_ckpt.with_suffix(".onnx")
                if not onnx_path.exists():
                    try:
                        self._export_to_onnx(module, onnx_path)
                    except Exception as e:
                        print(f"Export ONNX failed: {e}")
                    if self.use_mlflow and onnx_path.exists():
                        self.mlflow.log_artifact(str(onnx_path))
                if onnx_path.exists():
                    test_evaluator = CocoEvaluator(coco_gt=coco_gt, iou_types=self.evaluator.iou_types)
                    onnx_stats, _ = evaluate_onnx(
                        onnx_path,
                        self.test_dataloader,
                        test_evaluator,
                        self.device,
                        epoch=self.last_epoch,
                        use_mlflow=self.use_mlflow,
                    )
                    if self.use_mlflow and "coco_eval_bbox" in onnx_stats:
                        logs = {
                            f"test_onnx/{metric_names[idx]}": onnx_stats["coco_eval_bbox"][idx]
                            for idx in range(len(metric_names))
                        }
                        self.mlflow.log_metrics(logs, step=self.last_epoch)

        if self.use_mlflow:
            self.mlflow.end_run()

    def val(self):
        self.eval()
        if self.use_mlflow:
            self.mlflow.set_experiment(self.cfg.yaml_cfg.get("mlflow_experiment", "default"))
            self.mlflow.start_run(run_name=self.cfg.yaml_cfg.get("mlflow_run_name"))

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_mlflow=self.use_mlflow,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        if self.use_mlflow:
            self.mlflow.end_run()

        return
