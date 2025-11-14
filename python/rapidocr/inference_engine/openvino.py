# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig
from openvino.runtime import Core

from ..utils.download_file import DownloadFile, DownloadFileInput
from ..utils.log import logger
from .base import FileInfo, InferSession


class OpenVINOInferSession(InferSession):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        core = Core()

        model_path = cfg.get("model_path", None)
        if model_path is None:
            model_info = self.get_model_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            model_path = self.DEFAULT_MODEL_PATH / Path(model_info["model_dir"]).name
            download_params = DownloadFileInput(
                file_url=model_info["model_dir"],
                sha256=model_info["SHA256"],
                save_path=model_path,
                logger=logger,
            )
            DownloadFile.run(download_params)

        logger.info(f"Using {model_path}")
        model_path = Path(model_path)
        # self._verify_model(model_path)   # 因 OpenVINO IR(xml) 格式检测会失败，这里简便起见取消检测

        config = self._init_config(cfg)
        core.set_property("CPU", config)

        device = cfg.get("device", "CPU")   # 添加可能GPU情况
        precision_hint = cfg.get("inference_precision_hint", "f16")   # GPU默认会使用fp16，针对paddleocr模型这是OpenVINO推理时会发生的精度损失问题，所以暂用fp32精度进行推理来解决
        cache_dir = cfg.get("cache_dir", None)   # 加入模型缓存提高GPU二次推理速度
        gpu_config={"PERFORMANCE_HINT":"LATENCY","INFERENCE_PRECISION_HINT":precision_hint,"CACHE_DIR":cache_dir}   # 这里PERFORMANCE_HINT设置为延迟，还可以是THROUGHPUT，具体看这里(https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html)
        core.set_property("GPU", gpu_config)   # 设置GPU配置

        model_onnx = core.read_model(model_path)
        # compile_model = core.compile_model(model=model_onnx, device_name="CPU")
        compile_model = core.compile_model(model=model_onnx, device_name=device)   # 根据CPU/GPU选择相对应的config
        self.session = compile_model.create_infer_request()

    def _init_config(self, cfg: DictConfig) -> Dict[Any, Any]:
        config = {}
        engine_cfg = cfg.get("engine_cfg", {})

        infer_num_threads = engine_cfg.get("inference_num_threads", -1)
        if infer_num_threads != -1 and 1 <= infer_num_threads <= os.cpu_count():
            config["INFERENCE_NUM_THREADS"] = str(infer_num_threads)

        performance_hint = engine_cfg.get("performance_hint", None)
        if performance_hint is not None:
            config["PERFORMANCE_HINT"] = str(performance_hint)

        performance_num_requests = engine_cfg.get("performance_num_requests", -1)
        if performance_num_requests != -1:
            config["PERFORMANCE_HINT_NUM_REQUESTS"] = str(performance_num_requests)

        enable_cpu_pinning = engine_cfg.get("enable_cpu_pinning", None)
        if enable_cpu_pinning is not None:
            config["ENABLE_CPU_PINNING"] = str(enable_cpu_pinning)

        num_streams = engine_cfg.get("num_streams", -1)
        if num_streams != -1:
            config["NUM_STREAMS"] = str(num_streams)

        enable_hyper_threading = engine_cfg.get("enable_hyper_threading", None)
        if enable_hyper_threading is not None:
            config["ENABLE_HYPER_THREADING"] = str(enable_hyper_threading)

        scheduling_core_type = engine_cfg.get("scheduling_core_type", None)
        if scheduling_core_type is not None:
            config["SCHEDULING_CORE_TYPE"] = str(scheduling_core_type)

        logger.info(f"Using OpenVINO config: {config}")
        return config

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        try:
            self.session.infer(inputs=[input_content])
            return self.session.get_output_tensor().data
        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVIONError(error_info) from e

    def have_key(self, key: str = "character") -> bool:
        return False


class OpenVIONError(Exception):
    pass
