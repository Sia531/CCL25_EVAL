import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rich.progress import Progress
from tenacity import retry

from prompt import ARGUMENT_PROMPT, FRAME_PROMPT, ROLE_PROMPT

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ.get("AsyncOpenAI_API_KEY"), base_url=os.environ.get("BASE_URL")
)
semaphore = asyncio.Semaphore(4)

file_path = Path("./data/cfn-dataset/cfn-train.json")
frame_path = Path("./data/cfn-dataset/frame_info.json")

assert file_path.exists(), f"{file_path} not exist!"
assert frame_path.exists(), f"{frame_path} not exist!"


@dataclass
class Sample:
    ground_truth_frame: str  # 已知框架（训练集）
    word: str  # 目标词
    pos: str  # 词性
    data_id: int  # id
    text: str  # 文本内容
    prediction_frame: Optional[str] = None  # 预测框架


type Spans = list[list[int]]
type Spans_result = list[list[str]]


class SpansModel(BaseModel):
    content: Spans = Field(
        default_factory=list,
        description="多个span的list，每一个span有两个元素，span_begin_idx, span_end_idx",
    )


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
    frames = [frame["frame_name"] for frame in frame_data]
    return frames


def load_frame_entity_mappings(frame_path: Path) -> dict[str, list]:
    with open(file=frame_path, encoding="utf-8") as fp:
        frame_data = json.load(fp)
    mappings = {}
    for frame in frame_data:
        fes = frame["fes"]
        for fes_item in fes:
            fes_item.pop("fe_abbr")
            fes_item.pop("fe_ename")
        mappings[frame["frame_name"]] = fes
    return mappings


async def call_llm(messages: list[dict]) -> str:
    response = await client.chat.completions.create(
        model=os.environ.get("MODEL_NAME"),
        messages=messages,
        timeout=360,
        extra_body={"enable_thinking": True},
    )
    return response.choices[0].message.content


@retry
async def Frame_Identification(sample: Sample, frames: list[str]) -> str:
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
    pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern, text)
    result = match.group(1)
    sample.prediction_frame = result
    return result


@retry
async def Argument_Identification(sample: Sample) -> Spans:
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
                + f"允许思考，最终输出的Schema为{SpansModel.model_json_schema()}，按照这个输出对应的JSON，不要有其他内容",
            }
        ],
    )
    spans = SpansModel.model_validate_json(text)
    for item in spans.content:
        item.insert(0, sample.data_id)
    return spans.content


@retry
async def Role_Identification(
    sample: Sample, argument: Spans, frame_entity_mappings: dict[str, list]
) -> Spans:
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
        ],
    )
    result = json.loads(text)
    for idx, item in enumerate(argument):
        item.append(result[idx])
    return argument


async def Resolusion(
    sample: Sample,
    frames: list[str],
    frame_entity_mappings: dict[str, list],
) -> tuple[str, Spans, Spans]:
    async with semaphore:
        prediction_frame = await Frame_Identification(sample, frames)
        identification = await Argument_Identification(sample)
        prediction_role = await Role_Identification(
            sample, identification, frame_entity_mappings
        )
    return (prediction_frame, identification, prediction_role)


async def main():
    data = load_data(file_path=file_path)
    frames = load_frames(frame_path=frame_path)
    frame_entity_mappings = load_frame_entity_mappings(frame_path=frame_path)

    prediction_frame_list = []
    argument_identification_list = []
    role_identification_list = []
    tasks = []
    with Progress() as progress:
        task_id = progress.add_task("Task", total=len(data))
        for sample in data:
            tasks.append(
                asyncio.create_task(Resolusion(sample, frames, frame_entity_mappings))
            )
        for future in asyncio.as_completed(tasks):
            result: tuple[str, Spans, Spans] = await future
            prediction_frame_list.append(result[0])
            argument_identification_list.append(result[1])
            role_identification_list.append(result[2])
            progress.advance(task_id=task_id, advance=1)

    # wirte to json
    with open("data/submit/A_task1_test.json", "w") as task1:
        json.dump(prediction_frame_list, task1)
    with open("data/submit/A_task2_test.json", "w") as task2:
        json.dump(prediction_frame_list, task2)
    with open("data/submit/A_task3_test.json", "w") as task3:
        json.dump(prediction_frame_list, task3)


# result is list of tuple

if __name__ == "__main__":
    asyncio.run(main())
