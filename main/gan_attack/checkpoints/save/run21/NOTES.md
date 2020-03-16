Run 21: Overnight baseline run

| It | G accuracy | BB accuracy | D accuracy | BB accuracy (on G) | Comments |
|:------:|:------:|:------:|:------:|:------:|:-------:|
| 0      |     |       | 34.8%  |       |
| 1      | 78% |       | 36.8% | 76.0% |
| 2      | 80% | 53.00% | 55.5%  | 78.0% |
| 3      | 86% |        | 56.0% | 67.0% |
| 4      | 86% | 66.89% | 61.7%  | 64.0% |
| 5      | 79% |        | 64.6% | 76.0% |
| 6      | 96% | 77.61% | 61.6%  | 86.0% |
| 7      | 97% |        | 56.7% | 96.0% |
| 8      | 96% | 81.21% | 58.6%  | 91.0% |
| 9      | 96% |        | 58.2% | 93.0% |
| 10     | % | % | %  | % |
| 11     | % |        | % | % |
| 12     | % | % | %  | % |
| 13     | % |        | % | % |
| 14     | % | % | %  | % |
| 15     | % |        | % | % |
| 16     | % | % | % | % |
| 17     | % |        | % | % |
| 18     | % | % | % | % |
| 19     | % |        | % | % |
| 20     | % | % | % | % |
| 21     | % |        | % | % |
| 22     | % | % | % | % |
| 23     | % |        | % | % |
| 24     | % | % | % | % |
| 25     | % |        | % | % |
| 26     | % | % | % | % |
| 27     | % |        | % | % |
| 28     | % | % | % | % |
| 29     | % |        | % | % |
| 30     | % | % | % | % |
| 31     | % |        | % | % |
| 32     | % | % | % | % |
| 33     | % |        | % | % |

```
no_change_trigger = 5
nudge_trigger = 3
counter_nudge = True
fgsm = False
train_bb_every_it = 2
g_epochs = 8
d_refine_epochs = 200
reset_g_every_bb_train = False
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
{'epsilon': 0.5, 'norm': 'L1'}

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
 'refine_using_fgsm': False,
 'reset_g_every_bb_train': False,
 'train_bb_every_n_its': 2}
```