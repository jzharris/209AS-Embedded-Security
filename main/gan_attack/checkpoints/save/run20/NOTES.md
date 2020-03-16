Run 20: FGSM using L1 norm

| It | G accuracy | BB accuracy | D accuracy | BB accuracy (on G) | Comments |
|:------:|:------:|:------:|:------:|:------:|:-------:|
| 0      |     |       | 34.8%  |       |
| 1      | 79% |       | 37.7% | 69.0% |
| 2      | 79% | 53.17% | 58.1%  | 66.0% |
| 3      | 80% |        | 60.2% | 83.0% |
| 4      | 80% | 66.73% | 62.8%  | 69.0% |
| 5      | 80% |        | 63.0% | 82.0% |
| 6      | 89% | 76.96% | 52.1%  | 73.0% |
| 7      | 96% |        | 56.7% | 66.0% |
| 8      | 90% | 81.06% | 52.9%  | 72.0% |
| 9      | 94% |        | 54.1% | 63.0% | *
| 10     | 94% | 82.71% | 50.2%  | 78.0% |
| 11     | 96% |        | 39.3% | 77.0% |
| 12     | 97% | 86.28% | 51.2%  | 69.0% |
| 13     | 95% |        | 49.9% | 69.0% |
| 14     | 96% | 90.23% | 38.2%  | 81.0% |
| 15     | 88% |        | 52.1% | 92.0% |
| 16     | 95% | 91.84% | 40.2% | 78.0% |
| 17     | 92% |        | 40.3% | 69.0% | *
| 18     | 88% | 92.91% | 41.9% | 93.0% |
| 19     | 94% |        | 36.1% | 82.0% |
| 20     | 95% | 93.64% | 43.8% | 82.0% |
| 21     | 97% |        | 33.7% | 68.0% |
| 22     | 93% | 93.85% | 42.1% | 67.0% |
| 23     | 97% |        | 51.0% | 79.0% |
| 24     | 96% | 94.38% | 38.1% | 78.0% |
| 25     | 94% |        | 30.0% | 80.0% |
| 26     | 96% | 94.61% | 34.0% | 79.0% |
| 27     | 91% |        | 53.0% | 83.0% |
| 28     | 95% | 95.10% | 36.2% | 77.0% |
| 29     | 98% |        | 35.9% | 87.0% |
| 30     | 95% | 95.20% | 48.7% | 90.0% | *
| 31     | 94% |        | 37.2% | 85.0% |
| 32     | 96% | 95.37% | 50.6% | 83.0% | *
| 33     | 95% |        | 28.1% | 83.0% |

```
no_change_trigger = 5
nudge_trigger = 3
counter_nudge = True
fgsm = L1
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
 'refine_using_fgsm': True,
 'reset_g_every_bb_train': False,
 'train_bb_every_n_its': 2}
```