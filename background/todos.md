# TODO's

## Experiments

1. Perform label poisoning attack
    - vary the percentage of compromised clients/class
    
2. Correlation between estimator accuracy and the complexity of the estimator
    - hypothesis: InceptionV3 or FractalNet will perform well against most blackboxes
        - test against ~100 CNN configs and 2-3 datasets
        - use naive estimator during testing as a baseline
    
3. Perform GAN attack
    
4. Correlation between blackbox size and a change in the gradients during training
    - hypothesis: Increase in complexity/depth of model will result in more gradient decay

5. Use "Reverse Gradient" techniques to improve the training of the estimator
