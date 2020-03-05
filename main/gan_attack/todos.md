TODOs:

1. Create a second GAN, one that will try to maximize the loss of the BB model - which should
help our Discriminator converge to BB

** For some reason Step 2 has a bug where the test accuracy never changes at all, regardless
of training the discriminator...maybe something doesn't clear itself out? Bad loss?


[ ] Have GAN estimate multiple classes at once. This may help push the NN in the right direction,
given some shared features between different classes. (This requires a conditional GAN)

[ ] After training split learning, GAN needs the updated Discriminator (in EASY_MODE) or to refine the Discriminator o.w.
Right now, if Split Learning trains first, the GAN does not get an updated model. As simple as slg.gan.discriminator = _ ?

[ ] Add a loss and acc metrics to GAN D and G networks

[ ] Test the "better" label matching function

[ ] Remove dataset from GAN G training phase...not needed

[ ] Organize params

[ ] Reorganize local and global variables