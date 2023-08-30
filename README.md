# ForestCurveFit
TODO. in theory... CPU or optional GPU.js optimized curve fitter for scalar fields up to 4096 dimensions, such as training a small neuralnet near instantly. tensorflowjs was being slow and I just wanted it to do something small.

2023-8-30 <a href="https://github.com/benrayfield/ForestCurveFit/blob/main/earlyExperiments/ForestCurveFit003.html">earlyExperiments/ForestCurveFit003.html</a> is curve fitting using a 17 layer neuralnet that continuously changes the neuralActivationFunction per node among 20 such functions (see MathOp), WITHOUT BACKPROP. Instead of backprop, it uses calculus directly on the sum of squared loss for all weights and all training data at once. Its using 20 random data points, each a randon (y,x) position and random 1 (bright) vs -1 (dark) value, which are the white and black dots you see that it vibrates around. Its using neuralMomentum, but you can turn that off in the options js map. Theres also learnRate, velocityDecay, and other params. Its running on CPU so far, and I want to GPU optimize it in browser, but thats research. GPU.js doesnt support an array size 20 allocated in kernel, only float vars. I had originally planned to unroll the loop and use all scalar vars, but I dont know how fast GPU will do that since it makes the kernel code as big as the weights array. Maybe GLSL directly can do it (use code from <a href="https://github.com/benrayfield/jsutils/blob/master/src/TinyGLSLGraphicsEditor.html">https://github.com/benrayfield/jsutils/blob/master/src/TinyGLSLGraphicsEditor.html</a>), or maybe WebGPU (which works in Chrome and Brave browsers at least) can do that. TODO experiment with that. Its updating the screen about 5 times per second (5 FPS) which is a few times slower than https://playground.tensorflow.org/ that is supposedly using webgl glsl GPU in browser. Its unclear if this will be fast enough for live neural qlearning of the 3 balls, but need to reproduce tensorflow playground's behaviors on the spiral first.<br>
<img src="https://github.com/benrayfield/ForestCurveFit/blob/main/earlyExperiments/ForestCurveFit003.png?raw=true">

Some random models in <a href="https://github.com/benrayfield/ForestCurveFit/blob/main/earlyExperiments/ForestCurveFit002.html">earlyExperiments/ForestCurveFit002.html</a>. Its not curve-fitting yet but will be soon in CPU, and might have to make some changes to GPU optimize it but the plan is to make a UI similar to the graphics square on the right (but not the controls on the left) at https://playground.tensorflow.org/ and have the user put the include and exclude dots in the pic and watch it curvefit live. After thats working, maybe it will turn out to be fast enoufh for live neura qlearning of the 12d game of 3 bouncing balls?<br>
<img src="https://github.com/benrayfield/ForestCurveFit/blob/main/earlyExperiments/ForestCurveFit002.png?raw=true">

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


-----------------------<br><br>


TODO rewrite some of this text.


This is a variant of getThe10DataPointsWorkingWith4201DimensionalScalarFieldGpujsThenGoForNeuralQlearningOf3BouncingBalls. Do it this way cuz i can custom design, choose amounts of multiplies plusses sines sqrts etc, a func that can be computed really freakin fast in GPU cuz has very few dimensions and temporary vars. i wanna do it as scalar field of loss function over all the model weights, and combine that with the input output pairs that it should learn. it should learn the qscore of 12 dimensions to 1 dimension in case of my 3 bouncing balls. remember that all math ops, except maybe sqrt divide absval etc (cuz need to start positive) can be computed similar to mul(x,y)=tanh(atanh(x)*atanh(y)) allows mul(a,mul(b,c))=mul(mul(a,b),c) and keeps all numbers in range -1 to 1. Make this kind of neuralnet. try it with some simple data and simple models, then upgrade to neural qwlearning of 3 bouncing balls trying to raise one of the balls by moving one of the other balls so is similar to cartpole and other simple games like in openaiGym. Do this asap before get back to wikibinator.

Frustrated with tensorflowjs doing just around 100 megaflops when i was expecting 40 gigaflops....
screw this. im gonna make my own GPU optimized curve fitter. not a neuralnet specificly. just a really powerful browser GPU optimized curve fitter for forest of + * / sine arcsine exp log etc, with n input nodes, such as the 12 dimensions of yPosition xPosition yVelocity xVelocity for 3 bouncing balls i want to neural-qlearn on. my 15376 dimensional 4SAT solver is such a curve fitter. and im gonna pull that full teraflop from browser.
i dont need a neuralnet for sat solving since i defined a custom energy function that has lower energy the more SAT constraints are solved
you can play with it here. paint with 2 mouse buttons and rule110 grows on things https://memecombinator.io/experiments/ConvfieldDemo3.html

S:\q\q45x\w\forestcurvefit

if u use loss function as a scalar field of numModelWeights dimensions, then you cant overfit, but you can get stuck in localmin

my curve fitter will use a forest of up to 4096 flops (+ * sine arcsine exp log etc) so could contain very small neuralnets or arbitrary equations. gonna compile it to webgl shaders and have cpu kind too. tensorflowjs is gonna eat my dust. 32 bit opcodes. 12 bits for each of 2 pointers lower in an array, and 8 bits to choose which math op


i'll be able to learn some very simple functions in less than a millisecond

i think 1 teraflop should work ok. compile the whole up to 4096 dimensional scalar field to fit in a GPU core. run it on a bunch of starting positions in parallel. no backprop. calculus directly between the inputs and output. u change this input by epsilon, how much does the output change. up to 4096 times

up to 4096*32 bits. its 32 bits per opcode. its basically, each int tells it which 2 array indexs to read 2 scalars from, and the other 8 bits tells it what to do with them * + log sine etc, then write the result in current index

theres a 6 dimensional scalar field displayed as 3 bouncing balls (12d game state including positions and
velocities) in a html file, and the rule110 quasicrystal in another html file, since those are related
to the math. see earlyExperiments dir. i think i should get the thing working before taking pics of it.

theres a 6 dimensional scalar field displayed as 3 bouncing balls (12d game state
including positions and velocities) in a html file, and the rule110 quasicrystal
in another html file, since those are related to the math. see earlyExperiments dir.
i think i should get the thing working before taking pics of it.

i am gonna have so much fun with this thing. the little things it will allow me to do. like i could move stuff around a webpage in reaction to the mouse

if i get live neural-qlearning working even for very very simple games (such as 12-100 dimensional game states) thats likely to go as viral cuz ppl will play the games and make up new games involving qlearning into the rules themselves

2023-8-28[[ TODO
designWayInForestcurvefitToSimTheRule110QuasicrystalWithJustAFewTimesMoreDimsByHavingManySmallParallelThingsToAddToFormTotalEnergy
Also maybe should design way to fork n number of trainingData at once as loss func being
high dimensional scalar field.
Also remember that i plan to use this to make prototype of screwballscramble-like video games before
porting them or parts of th em to wikib, and there will be lots of sparse pieces in MMG of energyfunc
to sum together if it really is a huge 2d moving heightmap. And will use it for neuralqlearning of
3 bouncing balls, and more later once get that working. And for musical instruments to compute it
forward though not using that part for curvefitting.
Also, should it support 2d convolution (and 1d 3d 4d and what others?) vs have to duplicate
the int opcodes that many times, and this seems like another variant of running it n times
in parallel and adding those to model the loss function of training a neuralnet for n input/output vec pairs.
Also should this support other tensor ops such as matmul, that might be found in tensorflowjs,
or should I keep it simple and more limited to the few usecases thought of so far and similar patterns?
Those few usecases including simulating rule110 quasicrystal, painting with 2 mouse buttons
white and black pixels and fitting a mandelbrot fractal to it, curvfy lines in 2d bend around
obstacles as a puzzle game that AI can solve by curvefitting, neural-qlearning of at least 3 balls
at once to simulate airhockey and AI players learning to play it and optionally for more than
2 players at once adding more stuff to the game, musical instruments, etc.
Also, do I want normal backprop vs only the kind of calculus done on all weight dimensions at once?
TODO list the usecases, and the kinds of calculations andOr categories of those
(like tensor ops is a category) needed to make those usecases happen,
and choose a design before writing much more code.


Trying to organize this...
* given a vecN->vecM vecfield and T trainingData each a vecN and vecM, a way to only store the vecfield once but store the T*N inputs and EITHER (can choose either one, so this is really 2 or 3 ops) generate the T*M numbers OR generate a single number thats the sumOfSquaredError between the observed T*M numbers vs the given correct T*M numbers (what the inputs should generate near). This is an optimization to avoid duplicating the vecN->vecM again T times. Since its vecN (4 in the case of rule110 quasicrystal) to 1, these can be summed similar to how they're summed in the rule110 html quasicrystal demo, to adjust the velocity at each position (each pixel has a position and a velocity of its brightness) instead of having to do squared number of pixels the hard way (like will do it for neuralnets) but the rule110 html does it far more efficiently cuz its the sum of many energyfuncs.
* convolutional 1d. just for math completeness, even though i probably wont use this one since need at least 1d space 1d time to do anything useful.
* convolutional 2d. Example: rule110 quasicrystal. This is an optimization to avoid storing the vecN->num (energy func) centered on each pixel individually.
* convolutional 3d. Example: conwayLife quasicrystal. FIXME might want it to wrap around at weird offsets to make glider? Could do just the parts at the wrap the slower way?
* matmul AB BC.
* vecN->vecM small vectorField 
* sum of many energyfuncs, to be an energyfunc.

todo look thru the tensorflop ops of what u can do with multiple tensors to make another tensor, and consider which of those i want a similar thing to in this software.

What tensorflow is to a forest of tensors, ForestCurveFit (this software) is to to a forest of vectorfields. Combining vectorfields in various ways makes more vectorfields. In some cases by forking (similar to a loop but in parallel). Maybe in some cases by a sequential loop. In some cases by summing their outputs (like rule110 quasicrystal html sums potentialEnergy during computing calculus gradients), in some cases by concatting the outputs of 2 things resulting from same inputs. in some cases from concatting outputs and inputs so you just sum the quantity of each to get more outputs and more inputs, etc. This software should be able to run millions of sequential cycles per second, unlike I saw tensorflowjs do about 800 per second cuz its not designed for this kind of thing, or at least not in the webgl backend of it. I need at least 20,000 sequentially, times as much calculation per sound sample, to generate sound, for example. Tensorflow is functions making functions since the tensor view of it is lazyEvaled. But tensorflow wasnt designed for this kind of optimizing, and was not designed for what browser is fast at. It was designed as a desktop program first. Im designing from scratch to make it low lag and lots of CPU and GPU flops in browsers. And I think I can do it in alot smaller code than tensorflowjs is about 4mB nonminified and about 1mB minified. Im thinking closer to 50kB of code for the main code plus a few hundred kB for GPU.js optimization of it, plus the size of demos and experiments but you dont have to include those when you use it for other things.


Dont let this bunch of extra stuff it could also distract me from the main usecase of training very small neuralnets (like of 12 input nodes and 300 total weights) live in browser optimized by GPU.js, on a given  set of vecN->vecM output pairs and having a loss function for all of that. Has to be really small. If thats 300 dimensions, then it has to compute it 1+300 times in parallel, with 300 different directions of epsilon, to get the gradient.
Thats barely big enough to have any matmul in it. Maybe a few 10x10s, and raise size to 500 dims or 1000 dims. but thats about as big as i'd want it. maybe too big to use in realtime.
What I need is a proofOfConcept that I can play with (and id like to show ppl but mostly its for me, cuz i need to understand the efficiency etc of it before i can make a guess at how hard neuralQlearning will be).
So make a scalarfield for learning the spiral problem in https://playground.tensorflow.org/ and allow the user to put different include/exclude dots down with 2 mouse buttons and add those to training data. Just eval the model (whatever kind of model it is) on all the dots, and roll ball around the model weights. I think the problem is that neuralnet they're using only uses tanh but sine arcsine exp log / % + - etc (if tanh them and find a way to make them smooth or at least continuous position yet jagged in velocity etc) could model that better, AND using GPU it could explore alot more of the space 
...
https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=7,8,8,8,7,6&seed=0.22046&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false






see pic at https://twitter.com/benrayfield/status/1696288227072598377 and try the spiral at https://playground.tensorflow.org/ .

For my doubletrianglemodel, either choose manually which mathops go at which index, or add another array (by concat) of weights for each op, so for diag size n, theres n*n doubletriangle array plus n*numops weights (that each such row should sum to 1 when normed) so each diag can be all the ops continuously. but would have to solve the problem with sqrt, divide, log, etc needing positive input, unless just absvaled the input but that creats a sharp corner, or could square the input and sqrt it later. but that doesnt work for sqrt. in any case, i can put funcs very near those in. remember m(x,y)=tanh(atanh(x)*atanh(y)) leads to m(x,m(y,z))==m(m(x,y),z) and that can keep any such op in range -1 to 1.

For now manually choose the mathops per index while using doubletriangle array, and choose only the smooth ones. for example, only tanh and * and +1 etc or combined (1+tanh(sumA)*tanh(sumB).
Reproduce what tensorflowplayground did with the spiral. this shouldnt take long to write the code.
Make a html that displays it similarly, but just the part on the right with the 2 colors of dots to include and exclude, and let user paint more of those on there with 2 mouse buttons, and auto solve it. TODO.

Also I do want the n^2+n*numOps model, thats an upgrade of the doubletriangle model that does NOT need opcodes since it explores all possible opcodes and all possible forests of them. Maybe it has 3 ops each so its n*3*numOps?

