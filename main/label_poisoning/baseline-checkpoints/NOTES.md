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
| 10     | 96% | 82.97% | 51.0%  | 78.0% |
| 11     | 97% |        | 50.2% | 85.0% |
| 12     | 96% | 86.61% | 50.8%  | 100% | *
| 13     | 95% |        | 47.3% | 84.0% |
| 14     | 97% | 90.35% | 43.4%  | 89.0% |
| 15     | 96% |        | 47.3% | 80.0% |
| 16     | 96% | 92.00% | 52.0% | 89.0% |
| 17     | 95% |        | 45.2% | 93.0% |
| 18     | 94% | 92.96% | 38.2% | 91.0% |
| 19     | 92% |        | 37.4% | 94.0% |
| 20     | 97% | 93.52% | 39.8% | 91.0% |
| 21     | 96% |        | 38.4% | 95.0% | *
| 22     | 95% | 94.25% | 40.1% | 86.0% |
| 23     | 96% |        | 41.0% | 79.0% |
| 24     | 97% | 94.44% | 38.9% | 75.0% |
| 25     | 96% |        | 41.6% | 87.0% |
| 26     | 97% | 94.75% | 41.8% | 85.0% |
| 27     | 95% |        | 39.8% | 78.0% |
| 28     | 97% | 95.15% | 38.2% | 94.0% |
| 29     | 96% |        | 39.5% | 92.0% |
| 30     | 97% | 95.31% | 36.6% | 89.0% |
| 31     | 94% |        | 35.5% | 95.0% | *
| 32     | 96% | 95.61% | 39.5% | 79.0% |
| 33     | 95% |        | 37.3% | 100% | *

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