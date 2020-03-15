Run 19: FGSM using Inf norm - like in run18, but with preserving G!

| It | G accuracy | BB accuracy | D accuracy | BB accuracy (on G) | Comments |
|:------:|:------:|:------:|:------:|:------:|:-------:|
| 0      |     |       | 34.8%  |       |
| 1      | 78% |       | 38.4% | 79.0% |
| 2      | 78% | 52.77% | 56.9%  | 72.0% |
| 3      | 80% |        | 52.4% | 82.0% |
| 4      | 80% | 64.64% | 56.9%  | 78.0% |
| 5      | 80% |        | 57.8% | 68.0% |
| 6      | 91% | 76.46% | 62.8%  | 71.0% |
| 7      | 96% |        | 55.6% | 70.0% |
| 8      | 96% | 80.87% | 60.7%  | 89.0% |
| 9      | 96% |        | 62.1% | 83.0% | *
| 10     | 96% | 83.22% | 61.7%  | 84.0% |
| 11     | 91% |        | 54.7% | 73.0% |
| 12     | 98% | 87.08% | 62.3%  | 79.0% |
| 13     | 96% |        | 57.1% | 81.0% |
| 14     | 88% | 90.46% | 59.1%  | 92.0% | *
| 15     | 87% |        | 60.5% | 90.0% |
| 16     | 91% | 92.10% | 60.7% | 89.0% |
| 17     | 97% |        | 62.7% | 89.0% |
| 18     | 78% | 92.89% | 52.8% | 90.0% |
| 19     | 95% |        | 64.6% | 77.0% |
| 20     | 97% | 93.43% | 59.8% | 91.0% |
| 21     | 87% |        | 61.6% | 91.0% |
| 22     | 97% | 93.94% | 65.1% | 90.0% |
| 23     | 94% |        | 55.7% | 87.0% |
| 24     | 97% | 94.39% | 54.7% | 91.0% |
| 25     | 97% |        | 56.2% | 85.0% |
| 26     | 95% | 94.69% | 54.0% | 91.0% |
| 27     | 95% |        | 58.3% | 86.0% |
| 28     | 95% | 95.10% | 61.8% | 86.0% |
| 29     | 93% |        | 65.7% | 98.0% |
| 30     | 96% | 95.09% | 59.1% | 100% | *
| 31     | 96% |        | 64.1% | 95.0% |
| 32     | 94% | 95.31% | 63.0% | 100% | *
| 33     | 96% |        | 54.1% | 96.0% |

```
no_change_trigger = 5
nudge_trigger = 3
counter_nudge = True
fgsm = Inf
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
 'reset_g_every_bb_train': False,
 'train_bb_every_n_its': 2}
```