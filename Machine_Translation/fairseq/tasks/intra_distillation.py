# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import json
import torch
import torch.nn.functional as F

from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II

from fairseq import metrics, models
from fairseq.data import encoders
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)

def X_loss(logits, pad_mask):
    pad_mask = pad_mask.view(-1)
    non_pad_mask = ~pad_mask
    dict_size = logits[0].size(-1)

    m = sum(logits) / len(logits)
    m = m.float().view(-1, dict_size)[non_pad_mask]

    kl_all = 0
    for l in logits:
        l = l.float().view(-1, dict_size)[non_pad_mask]
        d = (l-m) * (torch.log(l) - torch.log(m))
        kl_all += d.sum()
    return kl_all / len(logits)

def JS_loss(logits, pad_mask):
    pad_mask = pad_mask.view(-1)
    non_pad_mask = ~pad_mask
    dict_size = logits[0].size(-1)

    m = sum(logits) / len(logits)
    m = m.float().view(-1, dict_size)[non_pad_mask]

    kl_all = 0
    for l in logits:
        l = l.float().view(-1, dict_size)[non_pad_mask]
        d = l * (torch.log(l) - torch.log(m))
        kl_all += d.sum()
    return kl_all / len(logits)


@dataclass
class TranslationIntraDistillationConfig(TranslationConfig):
    alpha: float = field(
        default=5.0,
        metadata={"help": "weight of the consistency loss"},
    )
    adaptive_alpha: int = field(
        default=0,
        metadata={"help": "whether use adaptive consistency method"},
    )
    
    ## This is a lazy implementation
    max_updates_train: int = field(
        default=25000,
        metadata={"help": "whether use adaptive consistency method"},
    )

    temperature_q: float = field(
        default=5,
        metadata={"help": "alpha get 1 at the step of N/temperature_q"},
    )

    temperature_p: float = field(
        default=2,
        metadata={"help": "alpha get max at the step of N/temperature_p"},
    )

    num_iter: int = field(
        default=2,
        metadata={"help": "Number of times go through the model"},
    )

    div: str = field(
        default="X",
        metadata={
            "help": "type of divergence"
        },
    )



@register_task("translation_intra_distillation", dataclass=TranslationIntraDistillationConfig)
class Translation_Intra_Distillation(TranslationTask):
    """
    Translation task for Switch Transformer models.
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    The translation task provides the following additional command-line
    arguments:
    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    def __init__(self, cfg: TranslationIntraDistillationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def build_model(self, cfg):
        model = models.build_model(cfg, self)

        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def _get_loss(self, sample, model, criterion):
        assert hasattr(
            criterion, "compute_loss"
        ), "translation_thor task requires the criterion to implement the compute_loss() method"

        encoder_out = model.encoder(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
        )
        net_output = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_out,
            src_lengths=sample["net_input"]["src_lengths"],
        )
        loss, nll_loss = criterion.compute_loss(model, net_output, sample, reduce=True)

        logits = net_output[0].float()
        logits = F.softmax(logits, dim=-1)

        sample_size = (
            sample["target"].size(0) if criterion.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "selection": net_output[1].get("selection", None),
        }

        return loss, logits, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        losses, logits, logging_outputs = [], [], []

        for _ in range(self.cfg.num_iter):
            with torch.autograd.profiler.record_function("forward"):
                with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                    loss, logit, sample_size, logging_output = self._get_loss(sample, model, criterion)
                    losses.append(loss)
                    logits.append(logit)
                    logging_outputs.append(logging_output)

        pad_mask = sample["target"].eq(criterion.padding_idx)
        if self.cfg.div == "X":
            intra_distillation_loss = X_loss(logits, pad_mask)
        elif self.cfg.div == "JS":
            intra_distillation_loss = JS_loss(logits, pad_mask)
        else:
            raise ValueError("Wrong type of divergence! Only support X and JS")

        alpha = self._get_alpha(self.cfg.alpha, update_num, self.cfg.max_updates_train)
        loss = sum(losses)/len(losses) + intra_distillation_loss * alpha
        
        logging_output = {
            "loss": torch.tensor([log["loss"] for log in logging_outputs]),
            "nll_loss": torch.tensor([log["nll_loss"] for log in logging_outputs]),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "intra_distillation_loss": intra_distillation_loss.data
        }

        if ignore_grad:
            loss *= 0

        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        # follows the reduce_metrics() function in label_smoothed_cross_entropy.py
        loss_sum = sum(log.get("intra_distillation_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "intra_distillation_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    def _get_alpha(self, alpha, num_update, max_update):
        if num_update >= max_update / self.cfg.temperature_p or not self.cfg.adaptive_alpha or alpha <= 1:
            return alpha
        else:
            alpha = torch.tensor([alpha])
            gamma = torch.log(1/alpha) / torch.log(torch.tensor([self.cfg.temperature_p/self.cfg.temperature_q])) # log_(p/q)(1/alpha)
            cofficient = ( self.cfg.temperature_p**gamma * alpha * num_update ** gamma) / (max_update ** gamma)
            return cofficient.item()
