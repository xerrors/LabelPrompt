from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.utils import ModelOutput
@dataclass
class BaseModelOutputWithPastAndCrossAttentionsAndActValue(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    act_value: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsAndActValue(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    act_value: Optional[Tuple[torch.FloatTensor]] = None



@dataclass
class MaskedLMOutputAndActValue(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    act_value: Optional[Tuple[torch.FloatTensor]] = None