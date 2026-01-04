import gym
import turtle
import numpy as np

class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.t = None
        self.unit = 50
        self.max_x = 12
        self.max_y = 4

    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)
            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit)
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit)

            for i in range(1, self.max_x - 1):
                self.draw_box(i, 0, 'black')
            self.draw_box(self.max_x - 1, 0, 'yellow')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)


def get_user_action():
    action = input("Enter action (W for up, S for down, A for left, D for right): ")
    try:
        action = action
        if action not in ['W', 'A', 'S', 'D']:
            raise ValueError
    except ValueError:
        print("Invalid action. Please enter W, A, S or D.")
        return get_user_action()

    if action == 'W':
        return 0
    elif action == 'A':
        return 3
    elif action == 'S':
        return 2
    elif action == 'D':
        return 1
    
    return action

def get_action(S):
    '''
    Give a state S, and you will return the action you act at that state.

    Example:
    get_action(12): W       # This means you go up at state 12
    '''
    pass

def learn(Q_table, S, action, next_state, action_prime, reward):
    pass


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env)

    S = env.reset()
    # env.render()
    done = False

    action = get_action(S)

    total_reward = 0
    while not done:
        next_state, reward, done, _ = env.step(action)

        action_prime = get_action(next_state)

        learn(Q_table, S, action, next_state, action_prime, reward)

        done = done
        total_reward += reward
        print("You Total reward is", total_reward)
        # env.render()
    
    print(f"Game Over! Total reward: {total_reward}")
    env.close()