"""Vision 과제: EuroSAT RGB 위성 사진을 10종 토지 피복으로 분류한다 (MobileNetV3-Small 미세조정).

- 데이터: Hugging Face `timm/eurosat-rgb`(MIT, 64x64). train·validation은 그대로 쓰고,
  test를 넷으로 나눠 t0(맑음) → t3(짙은 안개)로 갈수록 안개를 진하게 합성한다.
  센서나 날씨가 바뀌어 입력 분포가 변하는 상황(공변량 드리프트)을 흉내 낸다
- 내보내기: ONNX(model.onnx)와 전처리 정보(preprocess.json)
"""

import io
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from mlops.download import fetch_hf_dataset_file

logger = logging.getLogger(__name__)

REPO_ID = "timm/eurosat-rgb"
REVISION = "b4e28552cd5f3932b6abc37eb20d3e84901ad728"
SPLIT_FILES = MappingProxyType(
    {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
)
CLASS_NAMES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 시험 구간별 안개 세기. 0이면 원본
HAZE_LEVELS = MappingProxyType({"t0": 0.0, "t1": 0.25, "t2": 0.45, "t3": 0.65})
HAZE_COLOR = (225, 225, 230)
BLUR_RADIUS_PER_STRENGTH = 1.0
TEST_SHUFFLE_SEED = 0

INFERENCE_BATCH = 256
ONNX_INPUT = "image"
ONNX_OUTPUT = "logits"
ONNX_OPSET = 18


def apply_haze(image: Image.Image, strength: float) -> Image.Image:
    """밝은 회색과 섞고 살짝 흐리게 한다. 같은 입력이면 항상 같은 결과다."""
    if not 0.0 <= strength < 1.0:
        raise ValueError(f"haze 세기는 0 이상 1 미만이다: {strength}")
    rgb = image.convert("RGB")
    if strength == 0.0:
        return rgb
    hazy = Image.blend(rgb, Image.new("RGB", rgb.size, HAZE_COLOR), strength)
    return hazy.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS_PER_STRENGTH * strength))


def _open(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def decode_images(images: Iterable[bytes], image_size: int) -> np.ndarray:
    """이미지 바이트(PNG·JPEG)를 uint8 NHWC 배열로 바꾼다."""
    size = (image_size, image_size)
    arrays = []
    for data in images:
        image = _open(data)
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        arrays.append(np.asarray(image, dtype=np.uint8))
    if not arrays:
        return np.empty((0, image_size, image_size, 3), dtype=np.uint8)
    return np.stack(arrays)


def normalize(
    images: np.ndarray, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD
) -> np.ndarray:
    """uint8 NHWC → 정규화한 float32 NCHW (모델 입력)."""
    scaled = images.astype(np.float32) / 255.0
    normalized = (scaled - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))


def preprocess(
    images: Iterable[bytes],
    image_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    return normalize(decode_images(images, image_size), mean, std)


def _with_haze(df: pd.DataFrame, strength: float) -> pd.DataFrame:
    if strength == 0.0:
        return df.assign(haze=strength)
    hazed = [_png(apply_haze(_open(data), strength)) for data in df["image"]]
    return df.assign(image=hazed, haze=strength)


def build_slices(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """test를 겹치지 않게 HAZE_LEVELS 수만큼 나누고, 뒤 구간일수록 안개를 진하게 한다."""
    shuffled = test.sample(frac=1.0, random_state=TEST_SHUFFLE_SEED).reset_index(drop=True)
    periods = np.array_split(np.arange(len(shuffled)), len(HAZE_LEVELS))
    hazed = {
        name: _with_haze(shuffled.iloc[rows].reset_index(drop=True), strength)
        for (name, strength), rows in zip(HAZE_LEVELS.items(), periods, strict=True)
    }
    return {"train": train.assign(haze=0.0), "valid": valid.assign(haze=0.0), **hazed}


def _load_split(split: str) -> pd.DataFrame:
    raw = pd.read_parquet(fetch_hf_dataset_file(REPO_ID, SPLIT_FILES[split], REVISION))
    if not raw["label"].between(0, len(CLASS_NAMES) - 1).all():
        raise ValueError(f"{split}: 0~{len(CLASS_NAMES) - 1} 밖의 라벨이 있다")
    return pd.DataFrame(
        {
            "image_id": raw["image_id"],
            "image": raw["image"].map(lambda value: value["bytes"]),
            "label": raw["label"].astype("int64"),
        }
    )


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ImageNet으로 학습된 MobileNetV3-Small의 마지막 층만 클래스 수에 맞게 바꾼다."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = mobilenet_v3_small(weights=weights)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
    return net


@dataclass(frozen=True)
class VisionModel:
    net: nn.Module
    image_size: int


def _random_flip(x: torch.Tensor) -> torch.Tensor:
    """위성 사진은 방향이 의미가 없어서 좌우·상하 뒤집기로 데이터를 늘린다."""
    horizontal = (torch.rand(len(x)) < 0.5)[:, None, None, None]
    vertical = (torch.rand(len(x)) < 0.5)[:, None, None, None]
    x = torch.where(horizontal, x.flip(-1), x)
    return torch.where(vertical, x.flip(-2), x)


def _train_epoch(
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: np.ndarray,
    labels: torch.Tensor,
    batch_size: int,
) -> float:
    net.train()
    order = torch.randperm(len(labels))
    total = 0.0
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        x = _random_flip(torch.from_numpy(normalize(images[rows.numpy()])))
        loss = F.cross_entropy(net(x), labels[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(rows)
    return total / len(order)


def _in_batches(df: pd.DataFrame) -> Iterable[pd.Series]:
    for start in range(0, len(df), INFERENCE_BATCH):
        yield df["image"].iloc[start : start + INFERENCE_BATCH]


@dataclass(frozen=True)
class VisionTask:
    name: str = "vision"
    slices: tuple[str, ...] = ("train", "valid", *HAZE_LEVELS)
    train_slices: tuple[str, ...] = ("train",)
    valid_slices: tuple[str, ...] = ("valid",)
    max_eval_rows: int | None = None

    @property
    def default_params(self) -> Mapping[str, Any]:
        return {
            "epochs": 2,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "image_size": 64,
            # 0이면 학습 슬라이스 전부
            "max_train_samples": 0,
            "seed": 42,
        }

    @property
    def source(self) -> Mapping[str, Any]:
        return {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "license": "MIT",
            "haze_levels": dict(HAZE_LEVELS),
        }

    def prepare(self) -> dict[str, pd.DataFrame]:
        return build_slices(_load_split("train"), _load_split("validation"), _load_split("test"))

    def train(
        self, train: pd.DataFrame, valid: pd.DataFrame, params: Mapping[str, Any]
    ) -> VisionModel:
        torch.manual_seed(params["seed"])
        limit = params["max_train_samples"]
        if limit < 0:
            raise ValueError(f"max_train_samples는 0(전부) 이상이다: {limit}")
        if 0 < limit < len(train):
            train = train.sample(n=limit, random_state=params["seed"])
        images = decode_images(train["image"], params["image_size"])
        labels = torch.as_tensor(train["label"].to_numpy())

        model = VisionModel(net=build_model(len(CLASS_NAMES)), image_size=params["image_size"])
        optimizer = torch.optim.AdamW(
            model.net.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
        )
        for epoch in range(1, params["epochs"] + 1):
            loss = _train_epoch(model.net, optimizer, images, labels, params["batch_size"])
            accuracy = self.evaluate(valid, self.predict(model, valid))["accuracy"]
            logger.info("epoch %d: train loss %.4f, valid accuracy %.4f", epoch, loss, accuracy)
        return model

    def predict(self, model: VisionModel, df: pd.DataFrame) -> np.ndarray:
        model.net.eval()
        outputs = [np.empty(0, dtype=np.int64)]
        with torch.inference_mode():
            for images in _in_batches(df):
                x = torch.from_numpy(preprocess(images, model.image_size))
                outputs.append(model.net(x).argmax(dim=1).numpy())
        return np.concatenate(outputs)

    def evaluate(self, df: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
        return {"accuracy": float(np.mean(df["label"].to_numpy() == predictions)), "rows": len(df)}

    def export(self, model: VisionModel, out_dir: Path) -> None:
        net = model.net.eval()
        example = torch.zeros(2, 3, model.image_size, model.image_size)
        torch.onnx.export(
            net,
            (example,),
            str(out_dir / "model.onnx"),
            input_names=[ONNX_INPUT],
            output_names=[ONNX_OUTPUT],
            dynamic_shapes=({0: torch.export.Dim("batch")},),
            opset_version=ONNX_OPSET,
            external_data=False,
            dynamo=True,
            verbose=False,
        )
        spec = {
            "input": ONNX_INPUT,
            "output": ONNX_OUTPUT,
            "layout": "NCHW",
            "image_size": model.image_size,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "class_names": list(CLASS_NAMES),
        }
        (out_dir / "preprocess.json").write_text(json.dumps(spec, indent=2) + "\n")

    def predict_exported(self, out_dir: Path, df: pd.DataFrame) -> np.ndarray:
        import onnxruntime as ort

        spec = json.loads((out_dir / "preprocess.json").read_text())
        session = ort.InferenceSession(
            str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        outputs = [np.empty(0, dtype=np.int64)]
        for images in _in_batches(df):
            x = preprocess(images, spec["image_size"], spec["mean"], spec["std"])
            (logits,) = session.run([spec["output"]], {spec["input"]: x})
            outputs.append(logits.argmax(axis=1))
        return np.concatenate(outputs)


TASK = VisionTask()
