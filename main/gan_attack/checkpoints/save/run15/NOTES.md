Run 15: another baseline, but training BB more frequently

| It | G accuracy | BB accuracy | D accuracy |
|:------:|:------:|:------:|:------:|
| 0      |        | ~30%   | 34.8%  | 
| 1      | 78.0%  |        | 36.9%  | 
| 2      | 80.0%  | 53.21% | 55.5%  |
| 3      | 85.0%  |        | 54.0%  |
| 4      | 89.0%  | 67.58% | 58.6%  |
| 5      | 89.0%  |        | 55.9%  |
| 6      | 88.0%  | 77.69% | 51.2%  |
| 7      | 97.0%  |        | 51.6%  |
| 8      | 98.0%  | 80.90% | 51.0%  |
| 9      | 98.0%  |        | 44.2%  |
| 10     | 96.0%  | 83.08% | 47.1%  |
| 11     | 95.0%  |        | 42.9%  |

```
nudge = True
counter_nudge = True
fgsm = False
train_bb_every_it = 2
d_refine_epochs = 500 --> 200
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
 'd_refine_epoch_limit': 500,
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