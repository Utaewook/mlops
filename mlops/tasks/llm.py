"""LLM 과제: 한국어 뉴스 제목을 보고 주제(7종)를 생성한다 (Qwen3-0.6B-Base + LoRA).

- 데이터: KLUE YNAT(CC BY-SA 4.0). train과 validation을 합쳐 기사 연도(2016~2020)로 나눈다.
  2016~2018로 학습, 2019로 검증하고 2020은 미래 데이터다. 2020년은 사회 기사 비중이 크게 늘어서
  라벨 분포가 변하는 상황(사전 확률 드리프트)을 볼 수 있다.
  2020년 행은 모두 KLUE validation에서 온다. 같은 validation 안에서도 사회 비중이 2019년 17%에서
  2020년 68%로 뛰어서 시점에 따른 변화로 볼 수 있지만, 원본의 수집 방식 영향도 섞여 있을 수 있다
- Qwen3는 사전학습 때 이 시기 뉴스를 봤을 수 있다. 그래서 "미래"는 어댑터 입장의 미래이고,
  주로 보는 변화는 라벨 분포다. 라벨이 치우치면 정확도만으로는 부족해서 macro_f1을 같이 잰다
- 학습 슬라이스의 지표에는 학습에 쓴 행이 섞여 있어 낙관적이다.
  일반화 성능은 검증·미래 슬라이스로 본다
- 생성형이라 정해진 라벨 밖의 답을 낼 수 있다. 그 비율을 invalid_rate로 따로 잰다
- 내보내기: LoRA 어댑터, 토크나이저, 기본 모델 리비전과 프롬프트(serving.json)
"""

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mlops.download import fetch_hf_dataset_file

logger = logging.getLogger(__name__)

REPO_ID = "klue/klue"
REVISION = "63647dd911a8fcf76e560c17fef4c8bc009fbb24"
SPLIT_FILES = ("ynat/train/0000.parquet", "ynat/validation/0000.parquet")
LABELS = ("IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치")
INVALID_LABEL = "<invalid>"
SLICES = ("2016", "2017", "2018", "2019", "2020")

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
LORA_TARGETS = ("q_proj", "v_proj")

# 제목만 잘라서 길이를 맞춘다. 제목은 앞에 공백을 붙여 토큰화한다(이어진 문장과 같은 토큰이 되도록)
PROMPT_PREFIX = "다음 뉴스 제목의 주제를 고르세요.\n제목:"
PROMPT_SUFFIX = "\n주제:"
IGNORE_INDEX = -100
LOG_EVERY_STEPS = 10
# 원본 날짜는 두 형식이 섞여 있다: '2016.06.30. 오후 3:07'과 '2018.06.04 08:15'
KOREAN_DATE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\. (오전|오후) (\d{1,2}):(\d{2})$")
PLAIN_DATE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2}) (\d{2}):(\d{2})$")


def parse_date(text: str) -> pd.Timestamp:
    """'2016.06.30. 오후 3:07'이나 '2018.06.04 08:15'를 읽는다.

    오전 12시는 0시, 오후 12시는 12시다.
    """
    if korean := KOREAN_DATE.match(text):
        year, month, day, period, hour, minute = korean.groups()
        hour24 = int(hour) % 12 + (12 if period == "오후" else 0)
        return pd.Timestamp(int(year), int(month), int(day), hour24, int(minute))
    if plain := PLAIN_DATE.match(text):
        year, month, day, hour, minute = (int(part) for part in plain.groups())
        return pd.Timestamp(year, month, day, hour, minute)
    raise ValueError(f"날짜 형식이 다르다: {text!r}")


def split_by_year(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    years = df["published_at"].dt.year.astype(str)
    unknown = sorted(set(years) - set(SLICES))
    if unknown:
        raise ValueError(f"정의되지 않은 연도의 데이터가 있다: {unknown}")
    return {year: df[years == year].reset_index(drop=True) for year in SLICES}


def parse_label(generated: str) -> str:
    """생성한 글의 첫 줄이 라벨 중 하나면 그 라벨, 아니면 INVALID_LABEL."""
    first_line = generated.strip().split("\n", 1)[0].strip()
    return first_line if first_line in LABELS else INVALID_LABEL


def _ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_prompt(tokenizer: Any, title: str, max_prompt_tokens: int) -> list[int]:
    prefix, suffix = _ids(tokenizer, PROMPT_PREFIX), _ids(tokenizer, PROMPT_SUFFIX)
    budget = max_prompt_tokens - len(prefix) - len(suffix)
    if budget < 1:
        raise ValueError(
            f"max_prompt_tokens({max_prompt_tokens})가 "
            f"지시문({len(prefix) + len(suffix)}토큰)보다 커야 한다"
        )
    return prefix + _ids(tokenizer, " " + title)[:budget] + suffix


def _answer_ids(tokenizer: Any, label: str) -> list[int]:
    return _ids(tokenizer, " " + label) + [tokenizer.eos_token_id]


def build_example(
    tokenizer: Any, title: str, label: str, max_prompt_tokens: int
) -> tuple[list[int], list[int]]:
    """(input_ids, labels). 프롬프트 위치는 IGNORE_INDEX라서 정답과 EOS만 학습한다."""
    prompt = encode_prompt(tokenizer, title, max_prompt_tokens)
    answer = _answer_ids(tokenizer, label)
    return prompt + answer, [IGNORE_INDEX] * len(prompt) + answer


def _left_pad(rows: Sequence[Sequence[int]], value: int, width: int) -> torch.Tensor:
    return torch.tensor([[value] * (width - len(row)) + list(row) for row in rows])


def pad_prompts(prompts: Sequence[Sequence[int]], pad_id: int) -> dict[str, torch.Tensor]:
    """왼쪽을 채운다. 그래야 모든 행의 다음 토큰 위치가 맨 끝으로 맞춰진다."""
    width = max(len(p) for p in prompts)
    attention_mask = _left_pad([[1] * len(p) for p in prompts], 0, width)
    return {
        "input_ids": _left_pad(prompts, pad_id, width),
        "attention_mask": attention_mask,
        "position_ids": (attention_mask.cumsum(dim=1) - 1).clamp(min=0),
    }


def collate_left(
    examples: Sequence[tuple[Sequence[int], Sequence[int]]], pad_id: int
) -> dict[str, torch.Tensor]:
    batch = pad_prompts([ids for ids, _ in examples], pad_id)
    width = batch["input_ids"].shape[1]
    return {**batch, "labels": _left_pad([labels for _, labels in examples], IGNORE_INDEX, width)}


def answer_loss(model: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """정답 위치의 로짓만 계산한 교차 엔트로피. 전체 시퀀스로 구한 손실과 같다.

    어휘가 15만 개라 전체 위치의 로짓은 배치 하나에 수백 MB가 된다.
    왼쪽 패딩이라 정답은 각 행의 끝에 모여 있으므로,
    끝에서 k+1개 위치의 로짓만 받아 다음 토큰 k개를 맞힌다.
    """
    labels = batch["labels"]
    first_answer = (labels != IGNORE_INDEX).int().argmax(dim=1)
    k = int((labels.shape[1] - first_answer).max())
    if k >= labels.shape[1]:
        raise ValueError("정답 앞에 프롬프트가 있어야 한다")
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        logits_to_keep=k + 1,
    ).logits[:, :-1, :]
    targets = labels[:, -k:]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=IGNORE_INDEX
    )


def _pad_id(tokenizer: Any) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def generate_labels(
    model: Any, tokenizer: Any, titles: Sequence[str], max_prompt_tokens: int, batch_size: int
) -> np.ndarray:
    """탐욕 디코딩으로 라벨을 생성한다. 같은 가중치면 항상 같은 답이 나온다."""
    from transformers import GenerationConfig

    config = GenerationConfig(
        max_new_tokens=max(len(_answer_ids(tokenizer, label)) for label in LABELS),
        do_sample=False,
        pad_token_id=_pad_id(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
    )
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(titles), batch_size):
            prompts = [
                encode_prompt(tokenizer, t, max_prompt_tokens)
                for t in titles[start : start + batch_size]
            ]
            batch = pad_prompts(prompts, _pad_id(tokenizer))
            output = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                generation_config=config,
            )
            new_tokens = output[:, batch["input_ids"].shape[1] :]
            predictions.extend(
                parse_label(text)
                for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )
    return np.array(predictions, dtype=object)


def _load_base(repo_id: str, revision: str) -> Any:
    from transformers import AutoModelForCausalLM

    # CPU는 bf16 연산 가속이 없어서 fp32로 올린다(0.6B면 약 2.4GB)
    return AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, dtype=torch.float32)


@dataclass(frozen=True)
class LlmModel:
    model: Any
    tokenizer: Any
    base_model: str
    base_revision: str
    max_prompt_tokens: int
    eval_batch_size: int


def _train_steps(
    model: Any, examples: list[tuple[list[int], list[int]]], pad_id: int, params: Mapping[str, Any]
) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=params["learning_rate"])
    batch_size = params["batch_size"]
    model.train()
    step = 0
    for epoch in range(1, params["epochs"] + 1):
        order = torch.randperm(len(examples)).tolist()
        for start in range(0, len(order), batch_size):
            batch = collate_left([examples[i] for i in order[start : start + batch_size]], pad_id)
            loss = answer_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % LOG_EVERY_STEPS == 0:
                logger.info("epoch %d step %d: loss %.4f", epoch, step, loss.item())
    model.eval()


@dataclass(frozen=True)
class LlmTask:
    name: str = "llm"
    slices: tuple[str, ...] = SLICES
    train_slices: tuple[str, ...] = ("2016", "2017", "2018")
    valid_slices: tuple[str, ...] = ("2019",)
    # CPU에서 생성은 느리다. 슬라이스마다 고정 시드로 뽑은 표본만 평가한다
    max_eval_rows: int | None = 200

    @property
    def default_params(self) -> Mapping[str, Any]:
        return {
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "max_train_samples": 1000,
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "max_prompt_tokens": 48,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "eval_batch_size": 16,
            "seed": 42,
        }

    @property
    def source(self) -> Mapping[str, Any]:
        return {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "files": list(SPLIT_FILES),
            "license": "CC BY-SA 4.0",
        }

    def prepare(self) -> dict[str, pd.DataFrame]:
        raw = pd.concat(
            [pd.read_parquet(fetch_hf_dataset_file(REPO_ID, f, REVISION)) for f in SPLIT_FILES],
            ignore_index=True,
        )
        if not raw["label"].between(0, len(LABELS) - 1).all():
            raise ValueError(f"0~{len(LABELS) - 1} 밖의 라벨이 있다")
        df = pd.DataFrame(
            {
                "guid": raw["guid"],
                "title": raw["title"],
                "label": raw["label"].map(LABELS.__getitem__),
                "published_at": raw["date"].map(parse_date),
            }
        )
        return split_by_year(df.sort_values("published_at", kind="stable"))

    def train(
        self, train: pd.DataFrame, valid: pd.DataFrame, params: Mapping[str, Any]
    ) -> LlmModel:
        """LoRA로 학습한다. 조기 종료 없이 정해진 epoch만 돌고, valid 평가는 runner가 한다."""
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer

        torch.manual_seed(params["seed"])
        tokenizer = AutoTokenizer.from_pretrained(
            params["base_model"], revision=params["base_revision"]
        )
        lora = LoraConfig(
            r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            target_modules=list(LORA_TARGETS),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(_load_base(params["base_model"], params["base_revision"]), lora)
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "학습하는 파라미터: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total
        )

        sample = train.sample(
            n=min(params["max_train_samples"], len(train)), random_state=params["seed"]
        )
        examples = [
            build_example(tokenizer, title, label, params["max_prompt_tokens"])
            for title, label in zip(sample["title"], sample["label"], strict=True)
        ]
        _train_steps(model, examples, _pad_id(tokenizer), params)
        return LlmModel(
            model=model,
            tokenizer=tokenizer,
            base_model=params["base_model"],
            base_revision=params["base_revision"],
            max_prompt_tokens=params["max_prompt_tokens"],
            eval_batch_size=params["eval_batch_size"],
        )

    def predict(self, model: LlmModel, df: pd.DataFrame) -> np.ndarray:
        return generate_labels(
            model.model,
            model.tokenizer,
            df["title"].tolist(),
            model.max_prompt_tokens,
            model.eval_batch_size,
        )

    def evaluate(self, df: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import f1_score

        truth = df["label"].to_numpy()
        return {
            "accuracy": float(np.mean(truth == predictions)),
            "macro_f1": float(
                f1_score(truth, predictions, labels=list(LABELS), average="macro", zero_division=0)
            ),
            "invalid_rate": float(np.mean(predictions == INVALID_LABEL)),
            "rows": len(df),
        }

    def export(self, model: LlmModel, out_dir: Path) -> None:
        model.model.save_pretrained(out_dir / "adapter")
        model.tokenizer.save_pretrained(out_dir / "tokenizer")
        spec = {
            "base_model": model.base_model,
            "base_revision": model.base_revision,
            "labels": list(LABELS),
            "invalid_label": INVALID_LABEL,
            "prompt_prefix": PROMPT_PREFIX,
            "prompt_suffix": PROMPT_SUFFIX,
            "max_prompt_tokens": model.max_prompt_tokens,
            "eval_batch_size": model.eval_batch_size,
        }
        (out_dir / "serving.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")

    def predict_exported(self, out_dir: Path, df: pd.DataFrame) -> np.ndarray:
        from peft import PeftModel
        from transformers import AutoTokenizer

        spec = json.loads((out_dir / "serving.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(out_dir / "tokenizer")
        base = _load_base(spec["base_model"], spec["base_revision"])
        model = PeftModel.from_pretrained(base, out_dir / "adapter").eval()
        return generate_labels(
            model,
            tokenizer,
            df["title"].tolist(),
            spec["max_prompt_tokens"],
            spec["eval_batch_size"],
        )


TASK = LlmTask()
