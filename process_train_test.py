#!/usr/bin/env python
# coding: utf-8


import os
import random
import ujson
import json
import pandas as pd
import numpy as np
import torch

import time
import jsonlines
import copy
import requests

import openai
from openai import OpenAI

from constant import Constant
from collections import Counter

import argparse

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report





resistance_detection_prompt = """
# 角色：
你是一位非常专业的心理咨询师，你能够敏锐地察觉到来访者在心理咨询对话中产生的阻抗行为，并对来访者的阻抗行为进行准确的分类。

# 任务：
下面你将会被给到一段咨询师和来访者之间的心理咨询对话片段，包括上下文和来访者回应，以及来访者阻抗行为的分类体系。
请根据来访者阻抗行为分类体系，仔细品读上下文，判断来访者回应的唯一最合适的行为类型。

# 输入
## 来访者阻抗行为分类体系
1. 争辩：来访者质疑咨询师的专业性或诚信，对咨询师的资格或经验进行挑战性回应，对咨询师在咨询中所做的事的正确性表示怀疑，或认为咨询师不知道自己（来访者）在做什么，表达对咨询师的不满意。
争辩-挑战：来访者直接质疑咨询师所说内容、信息的准确性。
争辩-贬低：来访者质疑或直接贬低咨询师的个人能力、专业知识或在咨询中所起的作用。

2. 否认：来访者表现出不愿意认识问题、不合作、不愿意接受责任或采纳建议，无法配合咨询师的回应。
否认-责怪：当咨询师进行指导性的策略时，来访者以将问题归咎于其他人的方式来传达无法配合咨询师的意图。
否认-不认同：当咨询师使用指导性的策略时，例如引导来访者反思、引导来访者从新的视角看待问题、引导来访者思考可能的解决方案或直接提供建议等，来访者表示不认同咨询师，且未提供建设性的替代方案。
否认-找借口：来访者通过为自己的行为找借口，为自己辩解的方式，来表达不会遵从咨询师设定的方向。
否认-最小化：来访者暗示咨询师夸大了风险或危险，实际上情况并不那么糟糕。
否认-悲观：来访者对自己做出悲观、失败或消极的陈述，以此来表达不遵从咨询师设定的方向。
否认-犹豫：来访者对咨询师提供的信息或建议表示持保留态度。
否认-不愿改变：来访者表现出对现状的满意，缺乏改变的意愿；或表明不愿意改变。

3. 回避：来访者表现出对某个话题的逃避，不愿意讨论某个议题。
回避-最小的回应：来访者面对咨询师的开放式提问时，保留了重要信息，并以非常简短的、没有足够信息量的、没有体现出深度思考的回答进行回应，仅提供非常表面的信息予以比较敷衍的回复。
回避-界限设定：来访者直接拒绝或通过找各种理由来避免讨论某些话题。

4. 忽视：来访者表现出忽视或不跟随咨询师的迹象。在“忽视”行为下，来访者的回常常给人一种感觉为，没有跟随咨询师的指导，来访者似乎完全没有听到咨询师的话，或表现得好像咨询师什么也没说/问。
忽视-不关注：来访者沉浸在自己的话题或情绪中，他/她的回应仍在延续之前的陈述，表明他/她没有关注咨询师所说的话。
忽视-岔开话题：来访者改变咨询师所追求的谈话方向，提出新话题或关注点。

## 来访者合作行为分类体系
合作-无阻抗：来访者遵从咨询师设定的方向进行回应。

## 心理咨询对话：
### 上下文：
{0}

### 来访者回应：
{1}

# 输出格式
请输出一行，直接输出阻抗行为类别，不要输出任何其他内容。
阻抗行为必须为“争辩-挑战”“争辩-贬低”“否认-责怪”“否认-不认同”“否认-找借口”“否认-最小化”“否认-悲观”“否认-犹豫”“否认-不愿改变”“回避-最小的回应”“回避-理智化”“回避-界限设定”“忽视-不关注”“忽视-岔开话题”或“合作-无阻抗”中的一个。
"""


# # read labels
def load_defined_labels(label_filename):
    with open(label_filename, "r", encoding="utf-8") as f:
        defined_labels = ujson.load(f)
        defined_labels = [l["text"] for l in defined_labels]
    return defined_labels


def mapping_labels(defined_labels):
    Chinese_label_mapping = {}
    for label in defined_labels:
        coarse, fine = label.split("-")
        chinese_coarse, chinese_fine = coarse.split("/")[0], fine.split("/")[0]
        chinese_label = f"{chinese_coarse}-{chinese_fine}"
        Chinese_label_mapping[chinese_label] = label
    Chinese_label_mapping["合作-无阻抗"] = "合作/Collaboration-未阻抗/NA"
    return Chinese_label_mapping


def load_test_data(test_filename):
    with open(test_filename, "r", encoding="utf-8") as f:
        test_samples = ujson.load(f)
    return test_samples


def main(model_name, test_filename, label_mapping, output_filename, temp, topp):

    samples = load_test_data(test_filename)

    true_labels = {}
    prediction_answers = {}

    print(output_filename)
    if os.path.exists(output_filename):
        print("exist")
        with open(output_filename, "r", encoding="utf-8") as f:
            prediction_answers = ujson.load(f)
            prediction_answers = dict([(int(idx), answer) for idx, answer in prediction_answers.items()])

    print(len(prediction_answers))
    print("total samples = ", len(samples))

    for sample_id, sample in enumerate(samples):
        # print(sample_id)
        if sample_id not in prediction_answers or not prediction_answers[sample_id]:
            print(sample_id)
            print("ask llm")
            true_label = sample["label"]
            true_labels[sample_id] = true_label

            text = sample["text"]
            context_with_response = text.rsplit("\n\n", maxsplit=1)
            if len(context_with_response) == 2:
                context, response = context_with_response
            else:
                context = ""
                response = context_with_response[-1]

            message_text = resistance_detection_prompt.format(context, response)
            message = [{"role": "user", "content": message_text}]
            while True:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=message,
                    temperature=temp,
                    top_p=topp
                )
                prediction = completion.choices[0].message.content
                print(prediction)
                if prediction:
                    break

            prediction = label_mapping.get(prediction, prediction)
            prediction_answers[sample_id] = prediction

        with open(output_filename, "w", encoding="utf-8") as f:
            ujson.dump(prediction_answers, f, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process input name, output name, and model name server for generation.')
    parser.add_argument('--evaluation_type',
                        type=str,
                        help='test or valid',
                        default="test")
    parser.add_argument('--model_name', type=str, required=True, help='Model name used for generating responses')
    parser.add_argument('--fine_tune', type=str, required=True, help='fine_tune or base model')
    parser.add_argument('--host', type=str, required=True, help='host')
    args = parser.parse_args()

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{args.host}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    input_filename = f"data/all_{args.evaluation_type}_samples.json"

    label_filename = "../client-resistance-prompt/data/resistance_labels_for_doccano_2025-03-04.json"
    defined_labels = load_defined_labels(label_filename)
    Chinese_label_mapping = mapping_labels(defined_labels)

    temperature, topp = 0, 1.0

    for i in range(0, 3):

        # output_filename = f"prediction_results/{args.model_name.split('/')[-1]}_{args.fine_tune}_test_results_seed{i+120}_{temperature}_{topp}.json"
        if args.fine_tune:
            log_path = f"{args.fine_tune.rsplit('/', maxsplit=1)[0]}/{args.evaluation_type}_results"
            print(log_path)

            if not os.path.exists(log_path):
                os.mkdir(log_path)


            log_filename = f"{args.fine_tune.split('/')[-3]}_{args.fine_tune.split('/')[-2]}_{args.fine_tune.split('/')[-1]}"
            print(log_filename)
            output_filename = f"{log_path}/{log_filename}_test_results_seed{i}_{temperature}_{topp}.json"
            print(output_filename)
        else:
            output_filename = f"prediction_results/{args.model_name.split('/')[-1]}_prompt_test_results_seed{i+120}_{temperature}_{topp}.json"

        print(output_filename)
        main(args.model_name,
              input_filename,
              Chinese_label_mapping,
              output_filename,
              temperature,
              topp)











    


