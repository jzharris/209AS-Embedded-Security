TODOs:

[ ] After training split learning, GAN needs the updated Discriminator (in EASY_MODE) or to refine the Discriminator o.w.
Right now, if Split Learning trains first, the GAN does not get an updated model. As simple as slg.gan.discriminator = _ ?

[ ] Add a loss and acc metrics to GAN D and G networks

[ ] Test the "better" label matching function

[ ] Remove dataset from GAN G training phase...not needed