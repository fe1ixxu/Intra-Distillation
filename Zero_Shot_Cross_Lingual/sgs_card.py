# Given the loss, model, optimizer and oracle dataset, we show an example function of stochastic gradient surgery. 

import torch
def SGS_backward(args, loss, model, optimizer, oracle_datasets):
  if args.fp16:
    with amp.scale_loss(loss, optimizer) as scaled_loss:
      scaled_loss.backward()
  else:
    loss.backward()

  dev_langs = list(oracle_datasets.keys())
  grad_shapes = [p.shape if (p.requires_grad is True and p.grad is not None) else None
                   for group in optimizer.param_groups for p in group['params']]
  grad_numel = [p.numel() if (p.requires_grad is True and p.grad is not None) else 0
              for group in optimizer.param_groups for p in group['params']]
  grad = torch.cat([p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
                else torch.zeros(p.numel(), device=args.device) for group in optimizer.param_groups for p in group['params']], dim=0)
  model.zero_grad()

  def get_random_batch():
    oracle_lang = random.choice(dev_langs)
    length = len(oracle_datasets[oracle_lang])
    random_ind = torch.randint(0, length, (1,)).item()
    logger.info("*****, {}, {}, {}".format(oracle_lang, random_ind, length))
    oracle_batch = tuple(t.to(args.device) for t in oracle_datasets[oracle_lang][random_ind] if t is not None)

    inputs = {"input_ids": oracle_batch[0],
      "attention_mask": oracle_batch[1],
      "labels": oracle_batch[3],}

    if args.model_type != "distilbert":
      # XLM and RoBERTa don"t use segment_ids
      inputs["token_type_ids"] = oracle_batch[2] if args.model_type in ["bert", "xlnet"] else None

    if args.model_type == "xlm":
      inputs["langs"] = oracle_batch[4]
      
    return inputs

  oracle_inputs = get_random_batch()
  oracle_loss = model(**oracle_inputs)[0]

  if args.fp16:
    with amp.scale_loss(oracle_loss, optimizer) as scaled_loss:
      scaled_loss.backward()
  else:
    oracle_loss.backward()

  oracle_grad = torch.cat([p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
                else torch.zeros(p.numel(), device=args.device) for group in optimizer.param_groups for p in group['params']], dim=0)
  model.zero_grad()

  inner_product = torch.sum(grad * oracle_grad)
  project_direction = inner_product / torch.sum(oracle_grad * oracle_grad)
  grad = grad - torch.min(project_direction, torch.zeros_like(project_direction, device=args.device)) * oracle_grad

  indices = [0, ] + [v for v in accumulate(grad_numel)]
  params = [p for group in optimizer.param_groups for p in group['params']]
  assert len(params) == len(grad_shapes) == len(indices[:-1])
  for param, grad_shape, start_idx, end_idx in zip(params, grad_shapes, indices[:-1], indices[1:]):
      if grad_shape is not None:
          param.grad[...] = grad[start_idx:end_idx].view(grad_shape)  # copy proj grad