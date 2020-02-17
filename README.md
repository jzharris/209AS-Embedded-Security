# 209AS-Embedded-Security
Special Topics in Circuits and Embedded Systems: Security and Privacy for Embedded Systems, Cyber-Physical Systems, and Internet of Things

Website: https://jzharris.github.io/209AS-Embedded-Security/

## TODO's

#### Experiments

1. Perform label poisoning attack
    - vary the percentage of compromised clients/class
    
2. ✅ Correlation between estimator accuracy and the complexity of the blackbox
    - hypothesis: InceptionV3 or FractalNet will perform well against most blackboxes
        - test against ~100 CNN configs and 2-3 datasets
        - use naive estimator during testing as a baseline
        
3. ✅ Correlation between the loss of the blackbox and the ability for us to estimate it
    - hypothesis: poorly trained blackbox models will be harder to estimate because their outputs will "blanket" a 
    large area of possibilities.
        - train estimator using the same network...
    
4. Perform GAN attack
    
5. Correlation between blackbox size and a change in the gradients during training
    - hypothesis: Increase in complexity/depth of model will result in more gradient decay

6. Use "Reverse Gradient" techniques to improve the training of the estimator