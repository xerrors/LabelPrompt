from ast import arg
import csv
import pickle
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, Dataset
import torch.nn.functional as F
import numpy as np
import torch
from collections import OrderedDict
from data.funcs import add_tag

from data.make_cloze import make_cloze

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
# 这就是包内引用吗
import json
import re

rng = random.Random()

from transformers import AutoTokenizer

keyword_files = ["keyword_train.txt", "keyword_dev.txt", "keyword_test.txt"]

def tokenize(text, tokenizer):
    # berts tokenize ways
    # tokenize the [unused12345678910]
    D = [f"[unused{i}]" for i in range(10)]
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t)-1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in D:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t, add_special_tokens=False)
            for tok in tokens:
                text += [tok]

    for idx, t in enumerate(text):
        if idx + 3 < len(text) and t == "[" and text[idx+1] == "[UNK]" and text[idx+2] == "]":
            text = text[:idx] + ["[MASK]"] + text[idx+3:]

    return text

n_class = 1
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, entity=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.entity = entity


class InputExampleSST2(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, entity=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeaturesSST2(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id


class InputExampleWiki80(object):
    """A single training/test example for span pair classification."""

    def __init__(self, guid, sentence, span1, span2, ner1, ner2, label):
        self.guid = guid
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
        self.label = label

class InputFeatures(object):
    """ A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, entity=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity = entity


class DataProcessor(object):
    """ Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, data_dir, a):
        super().__init__()
        self.data_dir = data_dir

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, text_c=None, label=label))
        return examples



class relossProcessor(DataProcessor): #bert_s
    def __init__(self, data_path="data", use_prompt=False):
        def is_speaker(a):
            a = a.split()
            return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()

        # replace the speaker with [unused] token
        def rename(d, x, y):
            d = d.replace("’","'")
            d = d.replace("im","i")
            d = d.replace("...",".")
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y

        self.D = [[], [], []]
        for sid in range(3):
            # 分成三个数据集
            with open(data_path + "/"+["train.json", "dev.json", "test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(36):
                        if k+1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    d, h, t = rename(' '.join(data[i][0]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                    prompt = f"what is the relation between {h} and {t} ? {t} is the [MASK] {h} ."
                    d = [
                        prompt + d,
                        h,
                        t,
                        rid,
                        t
                    ]
                    self.D[sid] += [d]
        print(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(36)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=data[i][0], text_b=data[i][1], label=data[i][3], text_c=data[i][2], entity=data[i][4]))

        return examples

# TAG: BertProcessor
class bertProcessor(DataProcessor): #bert_s
    def __init__(self, data_path="data", use_prompt=False):
        def is_speaker(a):
            a = a.split()
            return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()

        # replace the speaker with [unused] token
        def rename(d, x, y):
            d = d.replace("’","'")
            d = d.replace("im","i")
            d = d.replace("...",".")
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y

        self.D = [[], [], []]
        for sid in range(3):
            # 分成三个数据集
            with open(data_path + "/"+["train.json", "dev.json", "test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            sample_idx = 0
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(36):
                        if k+1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    d, h, t = rename(' '.join(data[i][0]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                    if use_prompt:
                        prompt = f"{h} is the [MASK] {t} ."
                    else:
                        prompt = f"what is the relation between {h} and {t} ?"
                    sample_idx += 1
                    d = [
                        prompt + d,
                        h,
                        t,
                        rid,
                    ]
                    self.D[sid] += [d]
        print(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(36)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=data[i][0], text_b=data[i][1], label=data[i][3], text_c=data[i][2]))

        return examples


class ptuneProcessor(DataProcessor): #bert_s
    def __init__(self, data_path="data", use_prompt=False, ptune_k=6):
        def is_speaker(a):
            a = a.split()
            return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()

        # replace the speaker with [unused] token
        def rename(d, x, y):
            d = d.replace("’","'")
            d = d.replace("im","i")
            d = d.replace("...",".")
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y

        self.D = [[], [], []]
        """ TODO: add new samples
        every sample if there is a trigger then mask trigger and replace the origin mask with right token,
        if no trigger in the sentence, random mask a word in the sentence and replace the origin mask with the right token.

        """
        for sid in range(3):
            # 分成三个数据集
            with open(data_path + "/"+["train.json", "dev.json", "test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            sample_idx = 0
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(36):
                        if k+1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    d, h, t = rename(' '.join(data[i][0]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                    unused_word = " ".join([f"[unused{i}]" for i in range(3, ptune_k+3)])
                    # st 3,4 ; ed 5,6
                    st = [f"[unused{i}]" for i in range(3,5)]
                    ed = [f"[unused{i}]" for i in range(5,7)]
                    # 789 as prompt
                    prompt = f"[sub] {st[0]} {h} {st[1]} [sub] [unused7] [unused8] [MASK] [unused9] [obj] {ed[0]} {t} {ed[1]} [obj]."

                    # for temp_i in range(10):
                    #     d = d.replace(f"speaker {temp_i}:", f"[speaker{temp_i}]")

                    sample_idx += 1
                    sample = [
                        prompt + d,
                        h,
                        t,
                        rid,
                    ]
                    self.D[sid] += [sample]
                    # multi labels, add more data in the training set
                    if i == 0:
                        for idx,trigger in enumerate(data[i][1][j]['t']):
                            if trigger != "":
                                label_token = f"[C{data[i][1][j]['rid'][idx]+1}]"
                                prompt = prompt.replace("[MASK]", label_token)
                                # first assume the model predict the same output in the trigger, ...
                                d = d.replace(trigger, "[MASK]", 1)
                        sample = [
                            prompt + d,
                            h,
                            t,
                            rid,
                        ]
                        self.D[sid] += [sample]
        print(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(36)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=data[i][0], text_b=data[i][1], label=data[i][3], text_c=data[i][2]))

        return examples

# TAG: WIKI80Processor
class wiki80Processor(DataProcessor):
    """Processor for the TACRED data set."""
    def __init__(self, data_path, use_prompt):
        super().__init__()
        self.data_dir = data_path

    @classmethod
    def _read_json(cls, input_file):
        data = []
        with open(input_file, "r", encoding='utf-8') as reader:
            all_lines = reader.readlines()
            for line in all_lines:
                ins = eval(line)  # json.loads() with bug
                data.append(ins)
        return data

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "val.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, negative_label="no_relation"):
        data_dir = self.data_dir
        """See base class."""
        with open(os.path.join(data_dir,'rel2id.json'), "r", encoding='utf-8') as reader:
            re2id = json.load(reader)
        return re2id

    # TAG: Data - Create examples
    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            sentence = example['token']
            examples.append(InputExampleWiki80(guid=None,
                            sentence=sentence,  # tokens e.g. ['But', 'that', 'spasm', 'of', ...]
                            # BUG: maybe some bugs here, I don't -1
                            span1=(example['h']['pos'][0], example['h']['pos'][1]), # e.g. (14, 16)
                            span2=(example['t']['pos'][0], example['t']['pos'][1]), # e.g. (35, 36)
                            ner1=None,
                            ner2=None,
                            label=example['relation'])) # e.g. '/people/person/nationality'
        return examples

def convert_examples_to_features_for_loss(examples, max_seq_length, tokenizer):
    print("#examples", len(examples))
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenize(example.text_a, tokenizer)
        tokens_b = tokenize(example.text_b, tokenizer)
        tokens_c = tokenize(example.text_c, tokenizer)

        # t_tokens = tokenize(example.entity, tokenizer)
        t_tokens = tokenizer(example.entity, add_special_tokens=False)["input_ids"]

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_b + ["[SEP]"] + tokens_c

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label

        len_t = len(t_tokens)
        normal_input_ids = input_ids[:]
        for idx, input_id in enumerate(input_ids):
            if idx + len_t < len(input_ids) and input_ids[idx:idx+len_t] == t_tokens:
                # [MASK] id = 103
                for j in range(len_t):
                    input_ids[j+idx] = 103

        # append 1 sample with 2 input
        features.append(
            [InputFeatures(
                input_ids=normal_input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                entity = t_tokens
            ),
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                entity = t_tokens
            )]
        )

    print('#features', len(features))
    return features

def convert_examples_to_features_normal(examples, max_seq_length, tokenizer):
    print("#examples", len(examples))
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenize(example.text_a, tokenizer)
        tokens_b = tokenize(example.text_b, tokenizer)
        tokens_c = tokenize(example.text_c, tokenizer)


        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_b + ["[SEP]"] + tokens_c


        inputs = tokenizer(
            example.text_a,
            example.text_b + tokenizer.sep_token + example.text_c,
            truncation="longest_first",
            max_length=max_seq_length,
            padding="max_length",
            add_special_tokens=True
        )

        # tokens = []
        # segment_ids = []
        # tokens.append("[CLS]")
        # segment_ids.append(0)
        # for token in tokens_a:
        #     tokens.append(token)
        #     segment_ids.append(0)
        # tokens.append("[SEP]")
        # segment_ids.append(0)

        # for token in tokens_b:
        #     tokens.append(token)
        #     segment_ids.append(1)
        # tokens.append("[SEP]")
        # segment_ids.append(1)

        # input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # # tokens are attended to.
        # input_mask = [1] * len(input_ids)

        # # Zero-pad up to the sequence length.
        # while len(input_ids) < max_seq_length:
        #     input_ids.append(0)
        #     input_mask.append(0)
        #     segment_ids.append(0)

        # assert(inputs['input_ids'] == input_ids), print(inputs['input_ids'])

        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length

        label_id = example.label

        if ex_index == 0:
            print(f"input_text : {tokens_a} {tokens_b} {tokens_c}")
            print(f"input_ids : {inputs['input_ids']}")
            print(f"token_type_ids : {inputs['token_type_ids']}")

        # inputs = {}
        # inputs['input_ids'] = input_ids
        # inputs['attention_mask'] = input_mask
        # inputs['token_type_ids'] = segment_ids

        # append 1 sample with 2 input
        features.append(
            InputFeatures(
                input_ids=inputs['input_ids'],
                input_mask=inputs['attention_mask'],
                segment_ids=inputs['token_type_ids'],
                label_id=label_id,
            )
        )

    print('#features', len(features))
    return features


# TAG: Data - Features
def convert_examples_to_features(examples, max_seq_length, tokenizer, args, rel2id, data_mode=None):
    """Loads a data file into a list of `InputBatch`s."""

    save_file = "./dataset/cached_wiki80.pkl"
    mode = "text"

    num_tokens = 0
    dbpedia_hitted = 0
    instances = []

    use_bert = "BertTokenizer" in tokenizer.__class__.__name__
    use_gpt = "GPT" in tokenizer.__class__.__name__
    vocab_words = list(tokenizer.vocab.keys())

    if "dbpedia" in args.debug4:
        try:
            with open(os.path.join(args.data_dir, "entity.json"), "r") as file:
                ent_des_pair = json.load(file)
        except:
            print("\033[33m[WARNING]\033[0m", "Can not found ent-des pair in", os.path.join(args.data_dir, "entity.json"))
            ent_des_pair = {}

    def get_description(entity, tags=None, sentence=None, only_des=False):
        if not ent_des_pair or not ent_des_pair.get(entity):
            return " ".join(sentence) if sentence else ""

        des = ent_des_pair[entity]
        des = re.sub('\W+', ' ', des).replace("_", '')
        des = re.sub(r'\s+', ' ', des)

        if only_des:
            return des

        des = add_tag(des, tags, entity)
        return " ".join(sentence) + tokenizer.eos_token + des

    assert not (use_bert and use_gpt), "model cannot be gpt and bert together"

    print("\n\033[32mLoading\033[0m", data_mode, '...')

    for (ex_index, example) in enumerate(examples):
        """ the relation between SUBJECT and OBJECT is . """

        if ex_index % 10000 == 0 and ex_index != 0 or ex_index == len(examples) - 1:
            print("Writing example %d of %d" % (ex_index+1, len(examples)))

        tokens = []

        if mode.startswith("text"):
            for i, token in enumerate(example.sentence):
                if i == example.span1[0]: tokens.append("[s]")
                if i == example.span2[0]: tokens.append("[o]")
                if i == example.span1[1]: tokens.append("[/s]")
                if i == example.span2[1]: tokens.append("[/o]")
                tokens.append(token)

        SUBJECT = " ".join(example.sentence[example.span1[0]: example.span1[1]])
        OBJECT = " ".join(example.sentence[example.span2[0]: example.span2[1]])
        SUBJECT_ids = tokenizer(" " + SUBJECT, add_special_tokens=False)['input_ids']
        OBJECT_ids = tokenizer(" " + OBJECT, add_special_tokens=False)['input_ids']

        special_token = tokenizer.all_special_tokens
        # TAG: Get Prompts
        if use_gpt:
            if args.CT_CL:
                assert len(tokenizer.tokenize("[T1]")) == 1
                prompt = f"[T1] [T2] [T3] [sub] {OBJECT} [sub] [T4] [obj] {SUBJECT} [obj] [T5] {tokenizer.cls_token}"
            else:
                prompt = f"The relation between [sub] {SUBJECT} [sub] and [obj] {OBJECT} [obj] is {tokenizer.cls_token} ."
        else:
            # add prompt [T_n] and entity marker [obj] to enrich the context.
            if args.use_template_words == 1:
                if args.use_pre_prompt == "long" or args.use_pre_prompt == "long_mask":
                    pre_prompt = "[T1]" * (args.len_first_stage_prompt + 1) + "[T2]" * (args.len_first_stage_prompt + 1)  # there is no diff between T1 & T2, they will be replaced later...
                    prompt = pre_prompt + f"[sub] {SUBJECT} [sub] {tokenizer.mask_token} [obj] {OBJECT} [obj] ."
                else:
                    prompt = f"[sub] {SUBJECT} [sub] {tokenizer.mask_token} [obj] {OBJECT} [obj] ."

            elif args.use_template_words == 2:
                prompt = f"subject [sub] {SUBJECT} [sub] is {tokenizer.mask_token} , object [obj] {OBJECT} [obj] is {tokenizer.mask_token},  relation is {tokenizer.mask_token} ."

            elif args.use_template_words == 3:
                prompt = f"The relation between [sub] {SUBJECT} [sub] and [obj] {OBJECT} [obj] is {tokenizer.mask_token} ."

            else:
                prompt = f"{SUBJECT} {tokenizer.mask_token} {OBJECT}."

        token_str = " ".join(tokens)

        # if args.debug4 == "concat_train_instances":
        #     token_str += tokenizer.sep_token + find_train_instances()

        if "dbpedia" in args.debug4:
            sub_des = get_description(SUBJECT, only_des=True)
            obj_des = get_description(OBJECT, only_des=True)

            extra = ""
            if "ex" in args.debug4:
                if obj_des: extra += tokenizer.sep_token + obj_des
                if sub_des: extra += tokenizer.sep_token + sub_des
            else:
                if obj_des and SUBJECT in obj_des:
                    obj_des = obj_des[:min(len(obj_des), obj_des.find(SUBJECT) + 20)]
                    extra += tokenizer.sep_token + obj_des
                if sub_des and OBJECT in sub_des:
                    sub_des = sub_des[:min(len(sub_des), sub_des.find(OBJECT) + 20)]
                    extra += tokenizer.sep_token + sub_des

            if extra:
                dbpedia_hitted += 1
                extra = add_tag(extra, ["[s]", "[/s]"], SUBJECT)
                extra = add_tag(extra, ["[o]", "[/o]"], OBJECT)
                token_str += extra



        if args.add_label_options != "false" and args.add_label_options != "calc_key_values":
            prompt = "".join([f"[C{i}]" for i in range(1, args.rel_num+1)]) + tokenizer.sep_token + prompt

        inputs = tokenizer(
            prompt,
            token_str,
            truncation="longest_first",
            max_length=max_seq_length,
            padding="max_length",
            add_special_tokens=True
        )

        if use_gpt: cls_token_location = inputs['input_ids'].index(tokenizer.cls_token_id)
        # find the subject and object tokens, choose the first ones
        sep = sub_st = sub_ed = obj_st = obj_ed = -1
        for i in range(len(inputs['input_ids'])):
            if sep == -1 and inputs['input_ids'][i] == tokenizer.eos_token_id:
                sep = i
            if sub_st == -1 and inputs['input_ids'][i:i+len(SUBJECT_ids)] == SUBJECT_ids:
                sub_st = i
                sub_ed = i + len(SUBJECT_ids)
            if obj_st == -1 and inputs['input_ids'][i:i+len(OBJECT_ids)] == OBJECT_ids:
                obj_st = i
                obj_ed = i + len(OBJECT_ids)

        assert sub_st != -1 and obj_st != -1

        num_tokens += sum(inputs['attention_mask'])
        if ex_index == 0 and data_mode == 'train':
            print(f"Label : {example.label}")
            print(f"Text  : {tokenizer.decode(inputs['input_ids'][:sum(inputs['attention_mask'])])}")

        x = OrderedDict()
        x['input_ids'] = inputs['input_ids']
        if use_bert: x['token_type_ids'] = inputs['token_type_ids']
        x['attention_mask'] = inputs['attention_mask']
        x['label'] = rel2id[example.label]
        if use_gpt: x['cls_token_location'] = cls_token_location
        x['so'] =[sub_st, sub_ed, obj_st, obj_ed, sep]

        # @xerrors: make_cloze
        if args.use_cloze:
            masked_ids, masked_lm_positions, masked_lm_labels = make_cloze(tokenizer, inputs['input_ids'],
                            pos=[sep, sum(inputs['attention_mask'])],
                            rng=rng, vocab_words=vocab_words, predefined_cloze=args.use_cloze)

            if ex_index == 0 and data_mode == 'train':
                print(f"mask  : {tokenizer.decode(masked_ids[:sum(inputs['attention_mask'])])}")

            x['masked_ids'] = masked_ids
            x['masked_lm_positions'] = masked_lm_positions
            x['masked_lm_labels'] = masked_lm_labels

        if args.use_pre_prompt:
            sub_des = get_description(SUBJECT, ["[s]", "[/s]"], tokens)
            obj_des = get_description(OBJECT, ["[o]", "[/o]"], tokens)

            if args.use_pre_prompt == "default":
                sub_prompt = f"{'[T1]' * args.len_first_stage_prompt} [s] {SUBJECT} [/s] is {tokenizer.mask_token}."
                obj_prompt = f"{'[T1]' * args.len_first_stage_prompt} [o] {OBJECT} [/o] is {tokenizer.mask_token}."
            elif args.use_pre_prompt == "long_mask":
                sub_prompt = f"[s] {SUBJECT} [/s] is {tokenizer.mask_token * args.len_first_stage_prompt}."
                obj_prompt = f"[o] {OBJECT} [/o] is {tokenizer.mask_token * args.len_first_stage_prompt}."
            else:
                sub_prompt = f"{'[T1]' * args.len_first_stage_prompt} [s] {SUBJECT} [/s]"
                obj_prompt = f"{'[T1]' * args.len_first_stage_prompt} [o] {OBJECT} [/o]"

            sub_inputs = tokenizer(sub_prompt, sub_des, truncation="longest_first", max_length=max_seq_length, padding="max_length", add_special_tokens=True)
            obj_inputs = tokenizer(obj_prompt, obj_des, truncation="longest_first", max_length=max_seq_length, padding="max_length", add_special_tokens=True)

            x["sub_ids"] = sub_inputs["input_ids"]
            x["sub_mask"] = sub_inputs["attention_mask"]
            x["obj_ids"] = obj_inputs["input_ids"]
            x["obj_mask"] = obj_inputs["attention_mask"]

            if ex_index == 0 and data_mode == 'train':
                print(f'sub_prompt : {tokenizer.decode(sub_inputs["input_ids"][:sum(sub_inputs["attention_mask"])])}')
                print(f'obj_prompt : {tokenizer.decode(obj_inputs["input_ids"][:sum(obj_inputs["attention_mask"])])}')

        instances.append(x)

    # with open(file=save_file, mode='wb') as fw:
    #     pickle.dump(instances, fw)
    # print('Finish save preprocessed data to {}.'.format( save_file))

    input_ids = [o['input_ids'] for o in instances]
    attention_mask = [o['attention_mask'] for o in instances]
    if use_bert: token_type_ids = [o['token_type_ids'] for o in instances]
    if use_gpt: cls_idx = [o['cls_token_location'] for o in instances]
    labels = [o['label'] for o in instances]
    so = torch.tensor([o['so'] for o in instances])

    if args.use_cloze:
        masked_ids = torch.tensor([o['masked_ids'] for o in instances])
        masked_lm_labels = torch.tensor([o['masked_lm_labels'] for o in instances])
        masked_lm_positions = torch.tensor([o['masked_lm_positions'] for o in instances])

    if args.use_pre_prompt:
        sub_ids = torch.tensor([o['sub_ids'] for o in instances])
        sub_mask = torch.tensor([o['sub_mask'] for o in instances])
        obj_ids = torch.tensor([o['obj_ids'] for o in instances])
        obj_mask = torch.tensor([o['obj_mask'] for o in instances])

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    if use_gpt: cls_idx = torch.tensor(cls_idx)
    if use_bert: token_type_ids = torch.tensor(token_type_ids)
    labels = torch.tensor(labels)

    print("Average #tokens: %.2f." % (num_tokens * 1.0 / len(examples)), f"{dbpedia_hitted} hitted of {len(examples)}.")
    if use_gpt:
        dataset = TensorDataset(input_ids, attention_mask, cls_idx, labels)
    elif use_bert:
        dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels, so)
    elif args.use_cloze:
        dataset = TensorDataset(input_ids, attention_mask, labels, so, masked_ids, masked_lm_positions, masked_lm_labels)
    elif args.use_pre_prompt:
        dataset = TensorDataset(input_ids, attention_mask, labels, so, sub_ids, sub_mask, obj_ids, obj_mask)
    else:
        dataset = TensorDataset(input_ids, attention_mask, labels, so)

    return dataset

class BartDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self, ):
        return len(self.features)


def convert_examples_to_features_for_bart(examples, max_seq_length, tokenizer, args, rel2id):
    """Loads a data file into a list of `InputBatch`s."""

    save_file = "./dataset/cached_wiki80.pkl"
    mode = "text"

    num_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    instances = []


    label_id_to_decode = {}
    with open(f"{args.data_dir}/temp.txt", "r") as file:
        total_entity = []
        for line in file.readlines():
            t = line.strip().split("\t")
            entity = tokenizer(" ".join([t[2],t[4],t[3]]).strip(), add_special_tokens=True).input_ids
            total_entity.append([2]+entity)

        for idx, entity in enumerate(total_entity):
            label_id_to_decode[idx] = entity




    use_bert = "BertTokenizer" in tokenizer.__class__.__name__
    use_gpt = "GPT" in tokenizer.__class__.__name__

    assert not (use_bert and use_gpt), "model cannot be gpt and bert together"

    if False:
        with open(file=save_file, mode='rb') as fr:
            instances = pickle.load(fr)
        print('load preprocessed data from {}.'.format(save_file))

    else:
        print('loading..')
        for (ex_index, example) in enumerate(examples):


            """
                the relation between SUBJECT and OBJECT is .

            """

            if ex_index % 10000 == 0:
                print("Writing example %d of %d" % (ex_index, len(examples)))

            tokens = []
            SUBJECT_START = "[subject_start]"
            SUBJECT_END = "[subject_end]"
            OBJECT_START = "[object_start]"
            OBJECT_END = "[object_end]"


            if mode.startswith("text"):
                for i, token in enumerate(example.sentence):
                    if i == example.span1[0]:
                        tokens.append(SUBJECT_START)
                    if i == example.span2[0]:
                        tokens.append(OBJECT_START)
                    # for sub_token in tokenizer.tokenize(token):
                    #     tokens.append(sub_token)
                    if i == example.span1[1]:
                        tokens.append(SUBJECT_END)
                    if i == example.span2[1]:
                        tokens.append(OBJECT_END)

                    tokens.append(token)

            SUBJECT = " ".join(example.sentence[example.span1[0]: example.span1[1]])
            OBJECT = " ".join(example.sentence[example.span2[0]: example.span2[1]])
            SUBJECT_ids = tokenizer(" "+SUBJECT, add_special_tokens=False)['input_ids']
            OBJECT_ids = tokenizer(" "+OBJECT, add_special_tokens=False)['input_ids']

            if use_gpt:
                if args.CT_CL:
                    prompt = f"[T1] [T2] [T3] [sub] {OBJECT} [sub] [T4] [obj] {SUBJECT} [obj] [T5] {tokenizer.cls_token}"
                else:
                    prompt = f"The relation between [sub] {SUBJECT} [sub] and [obj] {OBJECT} [obj] is {tokenizer.cls_token} ."
            elif "bart" in args.model_name_or_path:
                prompt = f"The relation between [sub] {SUBJECT} [sub] and [obj] {OBJECT} [obj] is {tokenizer.mask_token} ."
            else:
                # add prompt [T_n] and entity marker [obj] to enrich the context.
                if args.use_template_words:
                    prompt = f"[sub] {SUBJECT} [sub] {tokenizer.mask_token} [obj] {OBJECT} [obj] ."
                else:
                    prompt = f"{SUBJECT} {tokenizer.mask_token} {OBJECT}."


            if ex_index == 0 and mode == 'train':
                input_text = " ".join(tokens)
                print(f"input text : {input_text}")
                print(f"prompt : {prompt}")
                print(f"label : {example.label}")
            inputs = tokenizer(
                prompt,
                " ".join(tokens),
                truncation="longest_first",
                max_length=max_seq_length,
                padding="max_length",
                add_special_tokens=True
            )
            if use_gpt: cls_token_location = inputs['input_ids'].index(tokenizer.cls_token_id)

            # find the subject and object tokens, choose the first ones
            sub_st = sub_ed = obj_st = obj_ed = -1
            for i in range(len(inputs['input_ids'])):
                if sub_st == -1 and inputs['input_ids'][i:i+len(SUBJECT_ids)] == SUBJECT_ids:
                    sub_st = i
                    sub_ed = i + len(SUBJECT_ids)
                if obj_st == -1 and inputs['input_ids'][i:i+len(OBJECT_ids)] == OBJECT_ids:
                    obj_st = i
                    obj_ed = i + len(OBJECT_ids)

            assert sub_st != -1 and obj_st != -1


            num_tokens += sum(inputs['attention_mask'])


            if sum(inputs['attention_mask']) > max_seq_length:
                pass
                # tokens = tokens[:max_seq_length]
            else:
                num_fit_examples += 1

            x = OrderedDict()
            x['input_ids'] = inputs['input_ids']
            x['attention_mask'] = inputs['attention_mask']
            x['label'] = rel2id[example.label]
            x['labels'] = label_id_to_decode[x['label']][1:]

            instances.append(x)


        with open(file=save_file, mode='wb') as fw:
            pickle.dump(instances, fw)
        print('Finish save preprocessed data to {}.'.format( save_file))


    print("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    print("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), max_seq_length))

    dataset = BartDataset(instances)

    return dataset



def convert_examples_to_feature_sst2(examples, max_seq_length, tokenizer, args, rel2id):
    """Loads a data file into a list of `InputBatch`s."""

    save_file = "./dataset/cached_wiki80.pkl"
    mode = "text"

    num_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    instances = []


    if False:
        with open(file=save_file, mode='rb') as fr:
            instances = pickle.load(fr)
        print('load preprocessed data from {}.'.format(save_file))

    else:
        print('loading..')
        for (ex_index, example) in enumerate(examples):
            try:
                prompt = f"[T1] [T2] {tokenizer.mask_token} ."
                inputs = tokenizer(
                    example.text_a + prompt,
                    truncation="longest_first",
                    max_length=max_seq_length,
                    padding="max_length",
                    add_special_tokens=True
                )

                x = OrderedDict()
                x['input_ids'] = inputs['input_ids']
                x['attention_mask'] = inputs['attention_mask']
                if "roberta" not in args.model_name_or_path:
                    x['token_type_ids'] = inputs['token_type_ids']
                x['label'] = int(example.label)


                instances.append(x)

            except Exception as e:
                print(e)

        with open(file=save_file, mode='wb') as fw:
            pickle.dump(instances, fw)
        print('Finish save preprocessed data to {}.'.format( save_file))

    input_ids = [o['input_ids'] for o in instances]
    attention_mask = [o['attention_mask'] for o in instances]

    if "roberta" not in args.model_name_or_path:
        token_type_ids = [o['token_type_ids'] for o in instances]
        token_type_ids = torch.tensor(token_type_ids)
    labels = [o['label'] for o in instances]


    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    print("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    print("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), max_seq_length))
    if "roberta" not in args.model_name_or_path:
        dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    else:
        dataset = TensorDataset(input_ids, attention_mask, labels)

    return dataset



def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()

def get_dataset(mode, args, tokenizer, processor):

    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    else:
        raise Exception("mode must be in choice [trian, dev, test]")
    gpt_mode = "wiki80" in args.task_name
    if "wiki80" in args.task_name and "bart" not in args.model_name_or_path and "t5" not in args.model_name_or_path:
        # normal relation extraction task
        dataset = convert_examples_to_features(
            examples, args.max_seq_length, tokenizer, args, processor.get_labels(), data_mode=mode
        )
        return dataset
    elif "wiki80" in args.task_name:
        dataset = convert_examples_to_features_for_bart(
            examples, args.max_seq_length, tokenizer, args, processor.get_labels()
        )
        return dataset
    elif "sst" in args.task_name:
        dataset = convert_examples_to_feature_sst2(
            examples, args.max_seq_length, tokenizer, args, None
        )
        return dataset
    else:
        train_features = convert_examples_to_features_normal(
            examples, args.max_seq_length, tokenizer
        )


    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []
    entity_id = []

    for f in train_features:
        input_ids.append(f.input_ids)
        input_mask.append(f.input_mask)
        segment_ids.append(f.segment_ids)
        label_id.append(f.label_id)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.float)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


    return train_data


def collate_fn(batch):
    pass




processors = {"normal": bertProcessor, "reloss": relossProcessor , "ptune": ptuneProcessor, "wiki80": wiki80Processor,
    "sst-2": Sst2Processor
}