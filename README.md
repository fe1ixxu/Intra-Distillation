# Intra-Distillation
This is the repository of our paper "Models Need Intra-Distillation: The Importance of All Parameters".

## Preproduction
We consider three tasks in our paper. Please visit the corrsponding folder and follow the instruction to reproduce the results.
* [Machine_Translation](https://github.com/fe1ixxu/Intra-Distillation/tree/master/Machine_Translation)
* [Natural_Language_Understanding](https://github.com/fe1ixxu/Intra-Distillation/tree/master/Natural_Language_Understanding)
* [Zero_Shot_Cross_Lingual](https://github.com/fe1ixxu/Intra-Distillation/tree/master/Zero_Shot_Cross_Lingual)

## Model Card
Intra-Distillation is easy to implement, we here provide a model card for eaiser takeaway.

### X-divergence
Given K logits in a list `logits` and padding masking `pad_mask`, we have
```
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
```

### Adaptive Alpha
Given max `alpha`, current step `num_update`, max step `max_update`, `p` and `q`, we have:
```
def _get_alpha(alpha, num_update, max_update, p, q):
    if num_update >= max_update / p or alpha <= 1:
        return alpha
    else:
        alpha = torch.tensor([alpha])
        gamma = torch.log(1/alpha) / torch.log(torch.tensor([p/q])) # log_(p/q)(1/alpha)
        new_alpha = ( p**gamma * alpha * num_update ** gamma) / (max_update ** gamma)
        return new_alpha.item()
```
