TODOs:

1. Make adaptive G/D training triggers - based on whether the training is stalling for G or D
2. Add loop_times to training params


[ ] Have GAN estimate multiple classes at once. This may help push the NN in the right direction,
given some shared features between different classes. (This requires a conditional GAN)

[ ] After training split learning, GAN needs the updated Discriminator (in EASY_MODE) or to refine the Discriminator o.w.
Right now, if Split Learning trains first, the GAN does not get an updated model. As simple as slg.gan.discriminator = _ ?

[ ] Add a loss and acc metrics to GAN D and G networks

[ ] Test the "better" label matching function

[ ] Remove dataset from GAN G training phase...not needed

[ ] Organize params

[ ] Reorganize local and global variables