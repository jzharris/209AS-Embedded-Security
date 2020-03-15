Run 18: FGSM using Inf norm - but with fixed accum queries!!

| It | G accuracy | BB accuracy | D accuracy | BB accuracy (on G) |
|:------:|:------:|:------:|:------:|:------:|
| 0      |     |       | 34.8%  |       |
| 1      | 79% |       | 38.2% | 71.0% |
| 2      | 69% | 53.64% | 57.5%  | 13.0% |
| 3      | 70% |        | 54.1% | 66.0% |
| 4      | 79% | 66.10% | 49.5%  | 16.0% |
| 5      | 84% |        | 49.1% | 59.0% |
| 6      | 94% | 77.01% | 47.0%  | 5.00% |
| 7      | 80% |        | 47.3% | 72.0% |
| 8      | 96% | 80.85% | 46.1%  | 12.0% |
| 9      | 89% |        | 45.3% | 76.0% |
| 10     | 94% | 82.27% | 44.3%  | 15.0% |
| 11     | 91% |        | 54.4% | 79.0% |
| 12     | 97% | % | %  | % |
| 13     | % |        | % | % |
| 14     | % | % | %  | % |
| 15     | % |        | % | % |

| It | G accuracy | BB accuracy | D accuracy | BB accuracy (on G) |
|:------:|:------:|:------:|:------:|:------:|
| 0      |        | ~30%   | 34.8%  |       |
| 1      | 78.0%  |        | 38.9%  | 84.0% |
| 2      | 79.0%  | 53.37% | 57.4%  | 13.0% |
| 3      | 70.0%  |        | 55.3%  | 75.0% |
| 4      | 80.0%  | 66.86% | 49.0%  | 09.0% |
| 5      | 80.0%  |        | 54.1%  | 63.0% |
| 6      | 92.0%  | 76.70% | %  | 52.07% |
| 7      | .0%  |        | %  | % |
| 8      | .0%  | % | %  | % |
| 9      | .0%  |        | %  | % |
| 10     | .0%  | % | %  | % |
| 11     | .0%  |        | %  | % |
| 12     | .0%  | % | %  | % |
| 13     | .0%  |        | %  | % |
| 14     | .0%  | % | %  | % |
| 15     | .0%  |        |   | % |

```
no_change_trigger = 5
nudge_trigger = 3
counter_nudge = True
fgsm = Inf
train_bb_every_it = 2
g_epochs = 8
d_refine_epochs = 200
reset_g_every_bb_train = True
```

```
split_training_params:
{'apply_gradients_after': 20,
 'batch_limit': None,
 'ckpt_folder': 'blackbox_checkpoint',
 'end_id': 'split_end_model',
 'epochs': 1,
 'eval_batch_size': 256,
 'full_id': 'split_model',
 'middle_id': 'split_middle_model',
 'minibatch_size': None,
 'shuffle_clients': True,
 'start_id': 'split_start_model'}

cgan_training_params:
{'batch_size': 256,
 'batches_per_epoch': 100,
 'bb_ckpt_folder': 'blackbox_checkpoint',
 'counter_nudge': True,
 'd_ckpt_folder': 'discriminator_checkpoint',
 'd_priming_epoch_limit': 1000,
 'd_refine_epoch_limit': 200,
 'd_reset_percentage': 1.0,
 'd_restore_after_nudge': True,
 'd_trigger': 0.98,
 'early_stop_trigger': 5,
 'end_id': 'd_end_model',
 'epochs': 8,
 'extra_depth': 3,
 'full_id': 'd_model',
 'g_ckpt_folder': 'generator_checkpoint',
 'g_nudge_probability': 0.2,
 'g_nudge_trigger': 3,
 'g_trigger': 1.01,
 'loop_times': 0,
 'middle_id': 'd_middle_model',
 'minibatch_size': None,
 'noise_dim': 100,
 'reset_g_every_it': False,
 'save_best_g': False,
 'softmax_power': 2,
 'start_id': 'd_start_model',
 'stop_sensitivity': 0.02,
 'uncertain_loop_times': 1,
 'use_bb_ends': True,
 'use_blackbox': False}

fgsm_training_params:
{'epsilon': 0.5, 'norm': 'Inf'}

attack_params:
{'accumulate_g_queries': True,
 'attack_classes': [1],
 'attack_trigger': 0.8,
 'attacker_clients': 5,
 'attacks_per_epoch': 10,
 'cgan_query_every_n_its': 1,
 'd_refinement_batch_num': 3,
 'd_refinement_batch_size': 100,
 'flip_to': [7],
 'flush_g_queries_every_bb_train': False,
 'our_class': 0,
 'prime_by_ckpt': True,
 'prime_cgan_by_ckpt': False,
 'prime_exit_trigger': 1.0,
 'prime_first_iteration': True,
 'prime_trigger': 0.0,
 'refine_exit_trigger': 1.0,
 'refine_using_fgsm': True,
 'reset_g_every_bb_train': True,
 'train_bb_every_n_its': 2}
```