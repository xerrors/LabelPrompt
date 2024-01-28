import torch
import torch.nn as nn
import torch.nn.functional as F

class RelClassifier(nn.Module):
    def __init__(self, feature_in, class_num, hidden_size=256):
        super(RelClassifier,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_in, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, class_num)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self,x):
        x = self.mlp(x)
        x = self.dropout(x)
        return x


class PrePromptEncoder(torch.nn.Module):
    def __init__(self, pre_seq_len, hidden_size):
        super().__init__()
        self.prefix_projection = True
        self.prefix_hidden_size = 256
        self.dropout = torch.nn.Dropout(0.1)

        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, self.prefix_hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.prefix_hidden_size, hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            pre_prompt = self.trans(prefix_tokens)
        else:
            pre_prompt = self.embedding(prefix)

        # pre_prompt = self.dropout(pre_prompt)

        return pre_prompt


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)  # ce loss
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()