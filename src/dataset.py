import asyncio
import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from tenacity import before_sleep_log, retry

from prompt import ARGUMENT_PROMPT, FRAME_PROMPT, ROLE_PROMPT

# =======================
# Logging Configuration
# =======================

logging.basicConfig(
    level=logging.INFO,  # 改成 DEBUG 可以看到更多信息
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
        )
    ],
)

logger = logging.getLogger("cfn-pipeline")

console = Console()

# =======================

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ.get("AsyncOpenAI_API_KEY"),
    base_url=os.environ.get("BASE_URL"),
)

semaphore = asyncio.Semaphore(4)

file_path = Path("./data/cfn-dataset/cfn-test-A.json")
frame_path = Path("./data/cfn-dataset/frame_info.json")

assert file_path.exists(), f"{file_path} not exist!"
assert frame_path.exists(), f"{frame_path} not exist!"


# =======================
# Data Structures
# =======================


@dataclass
class Sample:
    ground_truth_frame: str
    word: str
    pos: str
    data_id: int
    text: str
    prediction_frame: Optional[str] = None


type Spans = list[list[int]]


class SpansModel(BaseModel):
    content: Spans = Field(default_factory=list)


# =======================
# Utils
# =======================


async def call_llm(messages: list[dict]) -> str:
    start = time.perf_counter()

    response = await client.chat.completions.create(
        model=os.environ.get("MODEL_NAME"),
        messages=messages,
        timeout=360,
        extra_body={"enable_thinking": False},
    )

    cost = time.perf_counter() - start
    logger.debug(f"[dim]LLM cost: {cost:.2f}s[/]")
    logger.debug(f"[dim]LLM response: {response.choices[0].message.content}[/]")
    return response.choices[0].message.content


def load_data(file_path: Path) -> list[Sample]:
    with open(file=file_path, encoding="utf-8") as fp:
        data = json.load(fp)

    post_data = []
    for item_dict in data:
        label = item_dict.pop("frame")
        word = item_dict["text"][
            item_dict["target"][0]["start"] : item_dict["target"][0]["end"] + 1
        ]
        pos = item_dict["target"][0]["pos"]
        data_id = item_dict["sentence_id"]

        post_data.append(
            Sample(
                ground_truth_frame=label,
                word=word,
                pos=pos,
                data_id=data_id,
                text=item_dict["text"],
            )
        )
    return post_data


def load_frames(frame_path: Path) -> list[str]:
    with open(file=frame_path, encoding="utf-8") as fp:
        frame_data = json.load(fp)
    return [frame["frame_name"] for frame in frame_data]


def load_frame_entity_mappings(frame_path: Path) -> dict[str, list]:
    with open(file=frame_path, encoding="utf-8") as fp:
        frame_data = json.load(fp)

    mappings = {}
    for frame in frame_data:
        fes = frame["fes"]
        for fes_item in fes:
            fes_item.pop("fe_abbr", None)
            fes_item.pop("fe_ename", None)
        mappings[frame["frame_name"]] = fes
    return mappings


# =======================
# Pipeline Stages
# =======================


@retry(before_sleep=before_sleep_log(logger, logging.WARNING))
async def Frame_Identification(sample: Sample, frames: list[str]) -> str:
    logger.info(f"[bold cyan]FRAME[/] {sample.data_id} start")

    text = await call_llm(
        [
            {
                "role": "user",
                "content": FRAME_PROMPT.format(
                    text=sample.text,
                    word=sample.word,
                    pos=sample.pos,
                    frames=frames,
                ),
            }
        ]
    )

    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if not match:
        raise ValueError("No boxed result found")

    result = match.group(1)
    sample.prediction_frame = result

    logger.info(f"[cyan]FRAME[/] {sample.data_id} → {result}")
    return result


@retry(before_sleep=before_sleep_log(logger, logging.WARNING))
async def Argument_Identification(sample: Sample) -> Spans:
    logger.info(f"[bold yellow]ARG[/] {sample.data_id} start")

    text = await call_llm(
        [
            {
                "role": "user",
                "content": ARGUMENT_PROMPT.format(
                    text=sample.text,
                    prediction_frame=sample.prediction_frame,
                    word=sample.word,
                    pos=sample.pos,
                )
                + f"允许思考，最终输出Schema为{SpansModel.model_json_schema()}，"
                "输出JSON并放入<answer></answer>中",
            }
        ]
    )

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if not match:
        raise ValueError("No <answer> block found")

    spans = SpansModel.model_validate_json(match.group(1))

    for item in spans.content:
        item.insert(0, sample.data_id)

    logger.info(f"[yellow]ARG[/] {sample.data_id} spans={len(spans.content)}")

    return spans.content


@retry(before_sleep=before_sleep_log(logger, logging.WARNING))
async def Role_Identification(
    sample: Sample,
    argument: Spans,
    frame_entity_mappings: dict[str, list],
) -> Spans:
    logger.info(f"[bold magenta]ROLE[/] {sample.data_id} start")

    text = await call_llm(
        [
            {
                "role": "user",
                "content": ROLE_PROMPT.format(
                    text=sample.text,
                    prediction_frame=sample.prediction_frame,
                    word=sample.word,
                    pos=sample.pos,
                    argument=argument,
                    entities=frame_entity_mappings[sample.prediction_frame],
                ),
            }
        ]
    )

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if not match:
        raise ValueError("No role answer block")

    result = json.loads(match.group(1))

    for idx, item in enumerate(argument):
        item.append(result[idx])

    logger.info(f"[magenta]ROLE[/] {sample.data_id} roles={len(result)}")

    return argument


# =======================
# Orchestrator
# =======================


async def Resolusion(
    index: int,
    sample: Sample,
    frames: list[str],
    frame_entity_mappings: dict[str, list],
):
    logger.info(f"[dim]START[/] {sample.data_id}")

    async with semaphore:
        prediction_frame = await Frame_Identification(sample, frames)
        identification = await Argument_Identification(sample)
        prediction_role = await Role_Identification(
            sample,
            copy.deepcopy(identification),
            frame_entity_mappings,
        )

    logger.info(f"[bold green]DONE[/] {sample.data_id}")

    return (
        index,
        (sample.data_id, prediction_frame),
        identification,
        prediction_role,
    )


# =======================
# Main
# =======================


async def main():
    data = load_data(file_path)
    frames = load_frames(frame_path)
    frame_entity_mappings = load_frame_entity_mappings(frame_path)

    prediction_frame_list = [None] * len(data)
    argument_identification_list = [None] * len(data)
    role_identification_list = [None] * len(data)

    tasks = [
        asyncio.create_task(Resolusion(i, sample, frames, frame_entity_mappings))
        for i, sample in enumerate(data)
    ]

    with Progress(console=console) as progress:
        task_id = progress.add_task("Processing", total=len(tasks))

        for future in asyncio.as_completed(tasks):
            result = await future
            index = result[0]
            prediction_frame_list[index] = result[1]
            argument_identification_list[index] = result[2]
            role_identification_list[index] = result[3]
            progress.advance(task_id)

    # flatten
    task2_res = [span for group in argument_identification_list for span in group]
    task3_res = [span for group in role_identification_list for span in group]

    with open("data/submit/A_task1_test.json", "w") as f:
        json.dump(prediction_frame_list, f)

    with open("data/submit/A_task2_test.json", "w") as f:
        json.dump(task2_res, f)

    with open("data/submit/A_task3_test.json", "w") as f:
        json.dump(task3_res, f)

    logger.info("[bold green]ALL TASKS COMPLETED[/]")


if __name__ == "__main__":
    asyncio.run(main())
