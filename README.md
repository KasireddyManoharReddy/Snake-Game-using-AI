Snake Game Overview:
The Snake game is a classic arcade-style game where the player controls a snake moving on a grid. 
The objective is to eat food items (usually represented by fruits) to grow longer while avoiding collisions with the walls of the grid or the snake's own body.
As the snake eats more food, it increases in length, making navigation more challenging.
The game ends if the snake collides with itself or the boundaries, and the score typically increases with each food item consumed.

//Training method used to train snake

Q-Learning Training Method:
Q-Learning is a reinforcement learning algorithm used to train agents for decision-making in environments with discrete actions and states, such as the Snake game.
In Q-Learning, the agent learns an optimal policy by iteratively exploring and exploiting actions based on rewards received from the environment.
Initially, the agent explores randomly to gather experiences, gradually shifting towards exploiting learned knowledge to maximize rewards.
Through updating Q-values—representing the expected cumulative rewards for state-action pairs—based on observed rewards, Q-Learning enables the agent to improve its decision-making over time.
This iterative process continues until the agent converges to an optimal policy, capable of navigating the Snake game effectively to maximize its score while avoiding collisions.
