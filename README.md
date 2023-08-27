# ForestCurveFit
TODO. in theory... CPU or optional GPU.js optimized curve fitter for scalar fields up to 4096 dimensions, such as training a small neuralnet near instantly. tensorflowjs was being slow and I just wanted it to do something small.

Once I get this working hopefully it will be fast enough for around a 300-dimensional scalar field
to approximate the Q-score function for neural-qlearning of a 12 dimensional game state
of yPosition xPosition yVelocity xVelocity of 3 bouncing balls,
where you move 1 ball a little similar to airhockey to hit another ball
that you are trying to keep one one side of the game board,
which I already have the physics of working based on 6d scalar field gradients (not including velocity).
Once I get the curve fitter working and fast enough, Id like to use it to solve a constant scalar field
of all possible small neuralnet weights, or even simpler models than neuralnets just a bunch of
sines exps logs +s *s etc with a few hundred model weights (go parallel to the input 12 dimensions),
so one scalar field navigates the set of all possible smaller simpler scalar fields of a certain smaller/simpler kind,
the kind of scalar field from 12d game state to Q-score so it can use calculus gradients
to know where to move the puck to win a simulated airhockey game.
And thats just to get started. More complex games, though still low dimensional, and we'll see where it goes.
