# q-learning-homework

For this assignment you will be implementing a tabular Q-Learning agent from scratch and also following a PyTorch tutorial to code up a DQN agent to play CartPole. 


## Part 0:

Set up a virtual environment (conda, pyvenv) if you wish. If using conda you can just run.
```
conda create --name rl_env python=3.9
conda activate rl_env
```
Then install gymnasium and pygame. You can run 
```
pip install "gymnasium[box2d,mujoco]" pygame
```


Gymnasium is the most up-to-date and supported version of Gym that you played with for Homework 1. There are a few small differences between Gymnasium and Gym, but they follow the same overall structure and Gym is deprecated so it’s useful to get familiar with Gymnasium. Here is a link to documentation that you should skim to remind yourself how interactions with the environment works at a high-level: 
https://gymnasium.farama.org/introduction/basic_usage/ 

You will also need PyTorch installed. You don’t need to have a GPU since the problems we’ll be studying are simple, but if you have access to one you can install the GPU version and it might speed things up a bit; however, RL is often more CPU bound than GPU bound so you should be fine either way. You can install PyTorch locally following the instructions here: 
https://pytorch.org/get-started/locally/ 


## Part 1: Tabular Q-Learning to Win Big at Blackjack!!

For part 1, we will be implementing a basic vanilla Q-Learning agent (no function approximation) to learn to win at the game of Blackjack. 
Before you start, read the description of the Blackjack environment on Gymnasium: https://gymnasium.farama.org/environments/toy_text/blackjack/ 
You can also find tutorials on how to play Blackjack online, but note that real-world Blackjack at has some extra rules and actions beyond what this environment allows. We assume all you can do is “hit” or “stick” (also called “stand”).
There are a couple versions. We will just use the default version corresponding to 
gym.make('Blackjack-v1', natural=False, sab=False)

Your goal is to implement a Tabular Q-Learning agent that will interact with the environment to play Blackjack and will keep track of Q-values using a lookup table. 

Before starting, reread the section on Q-Learning from Sutton and Barto: 
http://incompleteideas.net/book/ebook/node65.html 

If you get stuck, this is also good reference, but your implementation doesn’t have to be this complicated: 
https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/#sphx-glr-tutorials-training-agents-blackjack-tutorial-py

To show that your agent is learning, plot a learning curve where time is the x-axis and cumulative reward over trajectories is the y-axis. You should see a noisy but roughly monotonically improving performance curve. If you want a smoother curve you can periodically pause updates and run your learned policy (remember the policy is implicit in the Q-values so just taking argmax over Q-values gives you the current policy) over several episodes and average the rewards and plot these averages over time. Give a brief report on your results.


## Part 2: Landing on the Moon using DQN!
Next we will implement the DQN algorithm. 
https://github.com/dsbrown1331/advanced-ai/blob/main/readings/dqn.pdf 
You will be following along with this tutorial: 
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
Note that this tutorial uses a slightly more sophisticated version of DQN that improves on the original Nature paper by using a Huber loss instead of MSE and using soft rather than hard target updates.

Also note the above tutorial focuses on training a policy for CartPole, but your goal is to train DQN learn how a policy for Lunar Lander instead. Feel free to start with CartPole since it is easier and should work out of the box using the code provided in the above tutorial. You should only require minimal changes to make the code work for Lunar Lander, but make sure you understand what is happening. If you're confused, Generative AI, is usually pretty good at explaining code snippets. Try it out and see.

Before you start coding things up take some time to familiarize yourself with the Lunar Lander environment. 
https://gymnasium.farama.org/environments/box2d/lunar_lander/ 
Then use the included human teleoperation code in this repo `lunar_lander_play.py`. It’s pretty tricky to land. We will see if we can use RL to learn a policy that is better than you are. You may need to install pygame to get it to work with keyboard inputs: https://www.pygame.org/wiki/GettingStarted

Provide evidence that your policy learns and improves overtime. You don’t have to spend a ton of time tuning hyper parameters or run policy learning for a really long time. Don't worry about getting a perfect policy. You should see significant improvement over a random policy after a few minutes of training. Report the performance of your learned policy versus a random policy. Add some code to visualize your policy using the env.render() functionality in Gymnasium. How does it do? Can it perform better than you can? Briefly report on your findings and answer the above questions.

