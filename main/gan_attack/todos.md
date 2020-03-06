### TODOs

1. [x] Make it so we don't have to train BB every time we want to query
2. [x] Save G imgs every iteration and append them to the query
3. Create a second GAN, one that will try to maximize the loss of the BB model - which should
help our Discriminator converge to BB
    * have a measure of 'uncertainty', that doesn't depend on the right answer only how
      certain the D thinks the image is (we want D to be very certain of SOMETHING)
        * maybe use [Monte Carlo dropout](https://towardsdatascience.com/is-your-algorithm-confident-enough-1b20dfe2db08)?
    * NOT a conditional GAN - use original architecture for G, use the same D
4. [x] Print out the metrics displayed by the pbars, for when viewing the ipynb a second time
5. [x] Add labels to generate_images()
6. Output summary of iteration at end of each iter. Append current accs to lists and display at
end of each iteration for easy access
7. Add regularization/extra loss to encourage displaying nothing at all as opposed to noise
* [x] nudge inc is not resetting

### Latest notes

Using the best G seems to have actually caused D to eventually degrade. Although, it is odd that,
after a while, D degrades altogether on it's own. This occurred after BB was getting high accuracy.

This is likely due to the fact that we rely heavily on G guessing the correct classes near the
beginning so D can get some inputs besides our own assigned class. This is all well and good, but
maximizing the accuracy of G did not get us the best result, eventually G degraded which caused
our refinement of D to fall apart.

Things to change:
* Second GAN should help.
* We also should save all intermediate outputs of G. Because, who knows, one version of G might
have been bad in the beginning but might now perform very well on the new D. So we should save
all interm (stable) G's and their output.
* Also, do we need to prime BB every time? If BB is high enough acc then stop priming it.
    * Add dropout to D so that getting the 0.098 issue will be harder to do
* Only nudge the classes that aren't performing well??