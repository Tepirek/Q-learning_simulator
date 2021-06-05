
from random import choice


Q_table = {
    1: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    2: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    3: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    4: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},

    5: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    6: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    7: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    8: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},

    9: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    10: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    11: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    12: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    
    13: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    14: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    15: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    16: {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
}

next_state_movement = {
    1: {'up': -1, 'down': 5, 'left': -1, 'right': 2},
    2: {'up': -1, 'down': 6, 'left': 1, 'right': 3},
    3: {'up': -1, 'down': 7, 'left': 2, 'right': 4},
    4: {'up': -1, 'down': 8, 'left': 3, 'right': -1},

    5: {'up': 1, 'down': 9, 'left': -1, 'right': 6},
    6: {'up': 2, 'down': 10, 'left': 5, 'right': 7},
    7: {'up': 3, 'down': 11, 'left': 6, 'right': 8},
    8: {'up': 4, 'down': 12, 'left': 7, 'right': -1},

    9: {'up': 5, 'down': 13, 'left': -1, 'right': 10},
    10: {'up': 6, 'down': 14, 'left': 9, 'right': 11},
    11: {'up': 7, 'down': 15, 'left': 10, 'right': 12},
    12: {'up': 8, 'down': 16, 'left': 11, 'right': -1},

    13: {'up': 9, 'down': -1, 'left': -1, 'right': 14},
    14: {'up': 10, 'down': -1, 'left': 13, 'right': 15},
    15: {'up': 11, 'down': -1, 'left': 14, 'right': 16},
    16: {'up': 12, 'down': -1, 'left': 15, 'right': -1}
}

rewards = {
    1: 0,
    2: 0,
    3: 0,
    4: 500,

    5: -100,
    6: 0,
    7: 0,
    8: 0,

    9: 0,
    10: 0,
    11: -100,
    12: 0,

    13: 0,
    14: 0,
    15: 0,
    16: 0,
}

episodes = {}

# początkowy stan agenta
current_state = 13
# początkowa wartość discount factor
gamma = 0.99
# początkowa wartość learning rate
learning_rate = 0.1
# koszt wykonania "kroku"
step_cost = -1
# początkowy numer epizodu
episode = 1 
# True jeśli chcemy uruchomić symulację
# False jeśli chcemy ręcznie wprowadzać ruchy agenta
simulation = True
# Liczba epizodów do symulacji
episodes_threshold = 100000

def print_current_position(current_state):
    print("Agent's position")
    print("-------------------")
    for i in rewards:
        character = "_"
        if i == current_state:
            character = "+"
        if not i % 4:
            print("%4s" % (character))
        else: 
            print("%4s" % (character), end="")
    print("-------------------")

def print_environment():
    print("\nEnvironment (4 x 4)")
    print("-------------------")
    for i in rewards:
        if not i % 4:
            print("%4d " % (rewards[i]))
        else:
            print("%4d " % (rewards[i]), end="")
    print("-------------------")

def print_q_table():
    print("\n------------------------------------------------------------------")
    print("| %10s | %49s |" % ("", "                     ACTION                      "))
    print("------------------------------------------------------------------")
    print("| %10s | %10s | %10s | %10s | %10s |" % ("state", "up", "down", "left", "right"))
    print("------------------------------------------------------------------")
    for i in Q_table:
        print("| %10d | %10.5f | %10.5f | %10.5f | %10.5f |" % (i, Q_table[i]["up"], Q_table[i]["down"], Q_table[i]["left"], Q_table[i]["right"]))
    print("------------------------------------------------------------------")

def update_state(episode, current_state, action):
    next_state = next_state_movement[int(current_state)][action]
    if next_state == -1:
        return episode, current_state, f"invalid action({action}) for current state({current_state})", False
    reward = rewards[next_state] + step_cost
    B = reward + gamma * max(
        Q_table[next_state]["up"],
        Q_table[next_state]["down"],
        Q_table[next_state]["left"],
        Q_table[next_state]["right"],
        )
    A = Q_table[current_state][action]
    temporal_difference = B - A
    current_state_value = A + learning_rate * temporal_difference
    Q_table[current_state][action] = current_state_value
    if reward == -101 or reward == 499:
        add_to_episode(episode, {"current_state": current_state, "action": action, "reward": reward, "next_state": next_state, "done": True})
        episode += 1
        if reward == -101:
            return episode, 13, "Destination reached! Starting new simulation!", True
        else:
            return episode, 13, "Agent died! Starting new simulation!", True
    else:
        add_to_episode(episode, {"current_state": current_state, "action": action, "reward": reward, "next_state": next_state, "done": False})
        return episode, next_state, "Valid move!", False

def add_to_episode(episode, data):
    if episode not in episodes.keys():
        episodes[episode] = []
        episodes[episode].append(data)
    else:
        episodes[episode].append(data)

def print_episode(episode):
    for i in episodes[episode]:
        print("current_state = %2d | action = %5s | reward = %9.4f | next_state = %2d | done = %s" %(i["current_state"], i["action"], i["reward"], i["next_state"], i["done"]))

if __name__ == "__main__":
    print("Enter gamma: ", end="")
    gamma = float(input().replace(',', '.'))
    print("Enter learning rate: ", end="")
    learning_rate = float(input().replace(',', '.'))

    previous_state = current_state

    if simulation:
        print_environment()
        
    while True:
        if not simulation:
            print_environment()
            print_current_position(current_state)
            print("Current state: %d" % (current_state))
            print("Enter action ('up', 'down', 'left', 'right'): ", end="")
            action = str(input())
            episode, current_state, message, show_episodes = update_state(episode, current_state, action)
            if previous_state != current_state:
                print_q_table()
            if show_episodes:
                print_episode(episode - 1)
            print(f"\nInfo: %s" % (message))
            previous_state = current_state
        else:
            action = choice(['up', 'down', 'left', 'right'])
            episode, current_state, message, show_episodes = update_state(episode, current_state, action)
            previous_state = current_state
            if episode == episodes_threshold:
                print_q_table()
                break
