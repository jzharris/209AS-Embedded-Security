Run 22: uGAN "overnight"

| It | G accuracy | BB accuracy | D accuracy | BB accuracy (on G) | Comments |
|:------:|:------:|:------:|:------:|:------:|:-------:|
| 0      |     |       | 34.8%  |       |
| 1      | 78% |       | 37.0% | 73.0% |
| 2      | 80% | 53.14% | 56.0%  | 80.0% |
| 3      | 88% |        | 56.3% | 77.0% |
| 4      | 95% | 66.53% | 55.6%  | 75.0% |
| 5      | 96% |        | 56.3% | 84.0% |
| 6      | 97% | 77.61% | 53.1%  | 79.0% |
| 7      | 95% |        | 53.2% | 75.0% |
| 8      | 95% | 81.11% | 55.5%  | 69.0% |
| 9      | 97% |        | 53.0% | 85.0% | *
| 10     | 97% | 83.33% | 54.9%  | 83.0% |
| 11     | 96% |        | 52.9% | 84.0% |
| 12     | 97% | 87.46% | 48.7%  | 89.0% |
| 13     | 96% |        | 46.3% | 92.0% |
| 14     | 96% | 90.39% | 43.6%  | 75.0% |
| 15     | 94% |        | 44.7% | 74.0% |
| 16     | 96% | 91.86% | 43.3% | 91.0% | *
| 17     | 96% |        | 42.6% | 85.0% |
| 18     | 97% | 92.88% | 41.2% | 88.0% |
| 19     | 97% |        | 39.8% | 74.0% |
| 20     | 96% | 93.40% | 37.2% | 78.0% |
| 21     | 96% |        | 32.1% | 74.0% |
| 22     | 96% | 93.94% | 38.2% | 66.0% |
| 23     | 96% |        | 37.6% | 75.0% |
| 24     | 97% | 94.40% | 36.4% | 62.0% |
| 25     | 96% |        | 45.9% | 70.0% |
| 26     | 96% | 94.56% | 43.1% | 87.0% |
| 27     | 98% |        | 40.7% | 80.0% |
| 28     | 96% | 94.88% | 38.2% | 68.0% |
| 29     | 96% |        | 38.4% | 75.0% |
| 30     | 97% | 95.05% | 35.3% | 67.0% |
| 31     | 93% |        | 37.9% | 72.0% |
| 32     | 96% | 95.36% | 37.7% | 75.0% | *
| 33     | 96% |        | 40.2% | 63.0% |

```
no_change_trigger = 5
nudge_trigger = 3
counter_nudge = True
fgsm = False
ugan = True
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
 'refine_using_ugan': True,
 'reset_g_every_bb_train': False,
 'train_bb_every_n_its': 2}

# uGAN training params:
ugan_training_params = {
    'minibatch_size': split_training_params['minibatch_size'],
    'extra_depth': 3,                           # number of extra middle layers to put in the D of cGAN
    'start_id': 'x_start_model',                # start piece
    'middle_id': 'x_middle_model',              # middle piece
    'end_id': 'x_end_model',                    # end piece
    'full_id': 'x_model',                       # full model name
    'use_bb_ends': True,                        # whether to share the weights of the start and end piece from the BB model
    'is_conditional': True,                     # whether to use the cGAN or uGAN architecture
    'batch_size': 256,                          # number of images to generate from uG at once
    'noise_dim': 100,                           # noise vector for uG
    'epochs': 15,                               # number of epochs to train uGAN
    'd_ckpt_folder': "discriminator_checkpoint",# folder where to store the d checkpoints
    'bb_ckpt_folder': "blackbox_checkpoint",    # folder where the blackbox default ckpt is kept
    'g_ckpt_folder': "generator_checkpoint",    # folder where to store the g checkpoints
    'batches_per_epoch': 100,                   # number of batches to train on per epoch
    'loop_times': 0,                            # number of times to apply softmax -> onehot encoding
    'uncertain_loop_times': 1,                  # number to use in the uncertain_loss used by D
    'softmax_power': 2,                         # number used in softmax -> onehot encoding operation
    'early_stop_trigger': 5,                    # stop training early, if g_accuracy has not improved for X epochs
    'stop_sensitivity': 0.02,                   # "no improvement" is when the g_accuracy has not moved more than X% from prev
    'save_best_g': False,                       # whether to save the best G during training, or to just use the last one
    'reset_g_every_it': True,                   # whether to restore uG back to init at the end of Step 5 if not -> Step 6
}
```