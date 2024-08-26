import os
import json
from argparse import ArgumentParser
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIBase():
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def predict(self, message, stream=False):

        if isinstance(message, str):
            messages=[{"role": "user", "content": message}]
        else:
            messages = message

        if stream:
            return self._stream_response(messages)
        else:
            return self._get_response(messages)

    def _stream_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            yield chunk.choices[0].delta

    def _get_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message

class VLLM(OpenAIBase):
    def __init__(self, model_name=None):
        model_name = model_name or "vllm"
        api_key = os.getenv("VLLM_API_KEY")
        base_url = os.getenv("VLLM_API_BASE")
        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name)


prompt_template = """
You are a relation classification model. Given a sentence and an entity pair, you need to predict the relation between them.

GOAL: Predict the relation between the entity pair in the sentence.
OUTPUT: the relation between the entity pair in the sentence without explanation.

relations: {rels}

sentence: "{sentence}"
entity pair: "{entity_pair}"

---

The relation between the entity pair in the sentence is: (relation):
"""

prompt_template_with_example = """
You are a relation classification model. Given a sentence and an entity pair, you need to predict the relation between them.

GOAL: Predict the relation between the entity pair in the sentence.
OUTPUT: the relation between the entity pair in the sentence without explanation.

relations: {rels}

examples:
{examples}

---

sentence: "{sentence}"
entity pair: "{entity_pair}"
relation:
"""



def construct_input(data, rels, examples=None):

    rels = ', '.join(rels)
    sentence = " ".join(data['token'])
    entity_pair = f"`{data['h']['name']}` and `{data['t']['name']}`"

    if examples:
        examples_str = "\n"
        for sample in examples:
            sentence = " ".join(sample['token'])
            entity_pair = f"`{data['h']['name']}` and `{data['t']['name']}`"
            label = sample['relation']
            examples_str += f"sentence: {sentence}\nentity pair: {entity_pair}\nrelation: {label}\n\n"

        return prompt_template_with_example.format(rels=rels, examples=examples_str, sentence=sentence, entity_pair=entity_pair)

    return prompt_template.format(rels=rels, sentence=sentence, entity_pair=entity_pair)


def get_examples(path, n):
    with open(path, 'r') as f:
        examples = [json.loads(line) for line in f]

    import random
    random.shuffle(examples)
    return examples[:n]



# 添加 args 参数
def add_args(parser):

    parser.add_argument("--dataset", type=str, default="dataset/retacred", help="Dataset to use")
    parser.add_argument("--samples", type=str, default="dataset/retacred/k-shot/8-1/train.txt", help="Path to the dataset")
    return parser


def main():
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    print(args)

    with open(os.path.join(args.dataset, 'rel2id.json'), 'r') as f:
        rel2id = json.load(f)

    test_data = []
    with open(os.path.join(args.dataset, 'test.txt'), 'r') as f:
        for line in f:
            test_data.append(eval(line))

    vllm = VLLM()


    correct = 0
    pred = 0
    total = 0

    try:

        for i, data in enumerate(test_data):
            label = data['relation']
            examples = get_examples(args.samples, 16)
            inputs = construct_input(data, rel2id.keys(), examples)

            response = vllm.predict(inputs).content
            print(f"{i}/{len(test_data)} Label: {label}")
            print(f"Response: {response}\n")

            if label != "NA" and label != "None" and label != "no relation":
                total += 1

            if "NA" in response or "None" in response or "no relation" in response:
                continue

            pred += 1
            if label in response:
                correct += 1

    except KeyboardInterrupt:
        print("Interrupted")

    precision = correct / pred
    recall = correct / total
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

if __name__ == "__main__":
    main()
