import pandas as pd
import pytest

from mlops.tasks.llm import (
    IGNORE_INDEX,
    INVALID_LABEL,
    SLICES,
    answer_loss,
    build_example,
    collate_left,
    encode_prompt,
    pad_prompts,
    parse_date,
    parse_label,
    split_by_year,
)


class CharTokenizer:
    """글자 하나를 토큰 하나로 바꾸는 테스트용 토크나이저."""

    eos_token_id = 0
    pad_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2016.06.30. 오전 10:36", "2016-06-30 10:36"),
        ("2019.01.02. 오후 3:07", "2019-01-02 15:07"),
        ("2019.01.02. 오후 12:05", "2019-01-02 12:05"),
        ("2019.01.02. 오전 12:05", "2019-01-02 00:05"),
        ("2018.06.04 08:15", "2018-06-04 08:15"),
        ("2018.09.16 17:28", "2018-09-16 17:28"),
    ],
)
def test_parse_date_reads_both_source_formats(text, expected):
    assert parse_date(text) == pd.Timestamp(expected)


def test_parse_date_rejects_unknown_format():
    with pytest.raises(ValueError, match="날짜"):
        parse_date("2016-06-30 10:36")


def test_year_slices_cover_every_row():
    df = pd.DataFrame({"published_at": pd.to_datetime(["2016-01-01", "2018-05-05", "2020-12-31"])})

    slices = split_by_year(df)

    assert tuple(slices) == SLICES
    assert [len(slices[y]) for y in SLICES] == [1, 0, 1, 0, 1]


def test_year_slices_reject_unknown_years():
    df = pd.DataFrame({"published_at": pd.to_datetime(["2021-01-01"])})

    with pytest.raises(ValueError, match="2021"):
        split_by_year(df)


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        (" 경제", "경제"),
        ("스포츠\n제목: 다음", "스포츠"),
        (" 날씨", INVALID_LABEL),
        ("", INVALID_LABEL),
    ],
)
def test_parse_label_accepts_only_known_labels(generated, expected):
    assert parse_label(generated) == expected


def test_prompt_truncates_only_the_title():
    tokenizer = CharTokenizer()
    short = encode_prompt(tokenizer, "가", max_prompt_tokens=100)
    long = encode_prompt(tokenizer, "가" * 200, max_prompt_tokens=len(short) + 5)

    fixed = len(short) - 2  # " 가"를 뺀 지시문 길이
    assert len(long) == len(short) + 5
    assert long[: fixed - 4] == short[: fixed - 4]
    assert long[-4:] == short[-4:]


def test_prompt_budget_must_fit_the_instruction():
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        encode_prompt(CharTokenizer(), "가", max_prompt_tokens=3)


def test_example_trains_only_on_the_answer_and_eos():
    tokenizer = CharTokenizer()

    input_ids, labels = build_example(tokenizer, "제목", "경제", max_prompt_tokens=100)

    prompt = encode_prompt(tokenizer, "제목", max_prompt_tokens=100)
    answer = [ord(c) for c in " 경제"] + [tokenizer.eos_token_id]
    assert input_ids == prompt + answer
    assert labels == [IGNORE_INDEX] * len(prompt) + answer


def test_collate_pads_on_the_left_and_ignores_padding():
    batch = collate_left([([5, 6, 7], [IGNORE_INDEX, 6, 7]), ([8], [8])], pad_id=1)

    assert batch["input_ids"].tolist() == [[5, 6, 7], [1, 1, 8]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [IGNORE_INDEX, IGNORE_INDEX, 8]]
    assert batch["position_ids"].tolist() == [[0, 1, 2], [0, 0, 0]]


def tiny_random_qwen3():
    """내려받지 않고 만드는 작은 무작위 Qwen3. 패딩·손실 계산의 정확성만 본다."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    return Qwen3ForCausalLM(config).eval()


def test_answer_loss_equals_the_full_sequence_loss():
    model = tiny_random_qwen3()
    batch = collate_left(
        [([5, 6, 7, 8, 9], [IGNORE_INDEX] * 3 + [8, 9]), ([10, 11, 12], [IGNORE_INDEX, 11, 12])],
        pad_id=1,
    )

    full = model(**batch).loss

    assert answer_loss(model, batch).item() == pytest.approx(full.item(), rel=1e-5)


def test_left_padding_does_not_change_greedy_generation():
    from transformers import GenerationConfig

    model = tiny_random_qwen3()
    config = GenerationConfig(max_new_tokens=5, do_sample=False, pad_token_id=1, eos_token_id=0)
    short, long = [5, 6, 7], [10, 11, 12, 13, 14, 15, 16]

    def generate(prompts):
        batch = pad_prompts(prompts, pad_id=1)
        width = batch["input_ids"].shape[1]
        output = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            generation_config=config,
        )
        return output[:, width:].tolist()

    assert generate([short, long])[0] == generate([short])[0]
