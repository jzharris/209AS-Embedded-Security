This folder contains some runs that may be useful to have in the future.

* save_1

    * contains a first-cut of a GAN training scheme that seems to be working. The GAN quickly overfits
    when trained naively. It finds examples in the dataset that fool the Blackbox model - regardless
    of whether they represent the "actual" class. In other words, the Generator tries to "break" the
    Blackbox model instead of estimating the dataset. This is caused by the nature of our problem:
    because we do not have access to the data in the other classes, we cannot push the Generator
    in the right direction.
    
    * in order to solve this problem, I experimented with a couple of things:
    
        1. Every 200 training cycles, if the Discriminator's test accuracy is down to being
        80% of its original accuracy or less, I restore the Discriminator's weights to their
        original values prior to training the GAN.
        2. I played around with using different loops in the softmax -> onehot encoding approx.
        The best result was when loop_times=0, so no approximation. This could be due to the fact
        that the approx. would make the gradients vanish very rapidly.
        3. During each training step, if the accuracy of the Generator grows to at least 95%, I
        turn stop training the Generator and start training the Discriminator. If this accuracy
        raises above 95%, I stop training the Generator and begin training the Discriminator. This
        way, the Generator will overfit less since the Discriminator keeps it in check.
            * I found that turning one on and the other off and visa versa worked better than 
            training the Discriminator and Generator at the same time. This could be due to the fact
            that I started off with using pretrained generator weights, though.
            
    * Things that need to change:
        1. Original accuracy needs to only use the portion of the dataset that we have access to.
        Currently I use the entire dataset.
        2. Train from scratch to see if this training scheme will work.
        3. Test with a better-trained Blackbox model -- this should increase our results.