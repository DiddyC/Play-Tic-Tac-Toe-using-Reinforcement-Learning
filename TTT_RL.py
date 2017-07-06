# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import random
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Sequential,load_model
from sklearn.utils import shuffle
import sys
from keras.utils import plot_model
import pydot
from keras.models import model_from_json



# Board size
size_board = 3

# Win reward
WINNERS_REWARD = 1.
# Draw reward
DRAW_REWARD = 0.
# Ordinary action reward
ACTION_REWARD = 0.

# Reward discount factor
gamma = 0.8

# Initial exploration rate
init_epsilon = 1.0
# Final exploration rate
final_epsilon = .01
# #of training episodes to anneal epsilon to final epsilon
annealingEpsilonEpisodes = 5000

# Number of training episodes to run
training_episodes = 12000

# Number of training episodes to accumulate stats
stats_history_max = 100


def create_neuralNetwork(input_size,output_size,hiddenLayers,learningRate):
    '''
    create NN architecture
    :param input_size: size of input
    :param output_size: size of output
    :param hiddenLayers: # of neurons in the hidden layer
    :param learningRate: 
    :return: a NN model
    '''

    model = Sequential()
    model.add(Dense(hiddenLayers[0], kernel_initializer='lecun_uniform',
                    input_shape=(input_size,)  ,activation='relu'))

    model.add(Dense(output_size, kernel_initializer='lecun_uniform', activation='linear'))
    optimizer = Adam(lr=learningRate)
    model.compile(loss="mse", optimizer=optimizer,metrics=['accuracy'])
    # print model
    #model.summary()
    plot_model(model, to_file='model_NN.png')
    return model

def updateWeights(updated_state,updated_action,updated_y):
    '''
    train NN
    :param updated_state: state of board
    :param updated_action: action vector
    :param updated_y: new training outputs of network 
    :return: logs of NN (history etc.)
    '''

    X = []
    for state in updated_state:
        X.append(state.reshape(18, ))
    X = np.array(X)
    # get current outputs for given inputs
    curr_y = NN_model.predict(X, batch_size=len(X))

    # iterate over the outputs and update them based on action and reward
    for i, output in enumerate(curr_y):
        previousAction_l = updated_action[i].reshape(9, )
        currentAction_Ind = np.where(previousAction_l == 1)
        curr_y[i][currentAction_Ind] = [updated_y[i]]

    # shuffle to avoid over-fitting
    X_shuffled, y = shuffle(X, curr_y, random_state=42)
    # train the model on shuffled date
    logs = NN_model.fit(X_shuffled, y, validation_split=0.0, epochs=1, verbose=0)
    return logs

def check_board_wins(board):
    '''
    
    :param board: state of board
    :return: True if there's a win
    '''
    board = board.reshape(1,9)[0]
    for a,b,c in [(0,1,2), (3,4,5), (6,7,8), #rows
                  (0,3,6), (1,4,7), (2,5,8), #columns
                  (0,4,8), (2,4,6)]:          #diagonals
        if board[a] == board[b] == board[c] == True:
            return True
    return False

def check_draw(sx, so):
    """
    Check for draw.
    """
    return np.all(sx+so)

def train(NN_model):
    '''
    Train the NN
    '''

    # Setup exploration rate parameters
    epsilon = init_epsilon
    step_eps = (init_epsilon - final_epsilon) / annealingEpsilonEpisodes

    # X player state
    stateX_Player = np.empty([size_board, size_board], dtype=np.bool)
    # O player state
    stateO_Player = np.empty_like(stateX_Player)

    # Accumulated stats
    stats_history = []
    lossK = []
    accuK = []

    # X move first
    PlayerX_turn = True

    numberOfEpisode = 1

    while numberOfEpisode <= training_episodes:
        # Start new game training episode
        stateX_Player[:] = False
        stateO_Player[:] = False

        # "stack" containing the previous states for training
        previous_states = [(None, None, None), (None, None, None)]  # [(state, action, reward(a))]

        move_num = 1

        while True:
            # Observe the next state
            currentState = get_state(PlayerX_turn, stateX_Player, stateO_Player)
            # Get Q values for all actions
            q_values = NN_model.predict(currentState.reshape(1,18),batch_size=1)[0].reshape(3,3)
            # Choose action based on epsilon-greedy policy
            q_max_index, CurrentAction_index = find_max_qValue(q_values, stateX_Player, stateO_Player, epsilon)

            # Retrieve previous player state/action/reward (if present)
            previousState, previousAction, previousReward = previous_states.pop(0)

            if previousState is not None:
                # Calculate updated Q value
                y_previous = previousReward + gamma * q_values[q_max_index]
                # Apply equivalent transforms of the board
                previousState, previousAction = find_equivalent_states(previousState, previousAction)
                # train the network on last action and state and their equivalents
                _logs  = updateWeights(previousState,previousAction,[y_previous] * len(previousState))


            # Apply action to state and get current state and reward
            current_reward, stateX_Player, stateO_Player, terminal_state = apply_move(PlayerX_turn, stateX_Player,
                                                                                        stateO_Player, CurrentAction_index)
            # update the current action vector
            currentAction = np.zeros_like(stateX_Player, dtype=np.float32)
            currentAction[CurrentAction_index] = 1.

            if terminal_state:  # win or draw
                # reward for current player
                y_current = current_reward
                # previous opponent state/action/reward
                previousState, previousAction, previousReward = previous_states[-1]
                # opponent receives a discounted negative reward
                y_previous = previousReward - gamma * current_reward

                # Apply equivalent transforms
                currentState, currentAction = find_equivalent_states(currentState, currentAction)
                previousState, previousAction = find_equivalent_states(previousState, previousAction)

                # Update Q-NN network
                updated_state = currentState + previousState
                updated_action = currentAction + previousAction
                updated_y = [y_current] * len(currentState) + [y_previous] * len(previousState)
                logs  = updateWeights(updated_state,updated_action,updated_y)

                lossK.append(float(logs.history['loss'][0]))  # loss
                accuK.append(float(logs.history['acc'][0])) # accuracy

                # Play test game before next episode to accumulate stats
                game_length, win_x, win_o = test(NN_model)
                stats_history.append([win_x or win_o, game_length])
                break

            # Store state, action and its reward
            previous_states.append((currentState, currentAction, current_reward))

            # Next player's move
            PlayerX_turn = not PlayerX_turn
            # count number of moves. Indicator of model success
            move_num += 1

        # Anneal down epsilon after episode
        if epsilon > final_epsilon:
            epsilon -= step_eps

        # Process stats_history
        if len(stats_history) >= stats_history_max:
            mean_win_rate, mean_length = np.mean(stats_history, axis=0)
            mean_lossK = np.mean(lossK)
            mean_accuK = np.mean(accuK)
            print("episode: %d," % numberOfEpisode, "epsilon: %.5f," % epsilon,
                  "mean win rate: %.5f," % mean_win_rate, "mean length: %.5f," % mean_length,
                  "mean loss: %.5f" % mean_lossK)

            # re- init lists for next round
            stats_history = []
            lossK = []
            accuK = []

        # Next episode counter
        numberOfEpisode += 1

def test(NN_model):
    """
    Play test game.
    """
    # X player state
    stateX_Player = np.zeros([size_board, size_board], dtype=np.bool)
    # O player state
    stateO_Player = np.zeros_like(stateX_Player)

    PlayerX_turn = True
    move_num = 1


    while True: # play till the end of the game
        # get the state of the board
        currentState = get_state(PlayerX_turn, stateX_Player, stateO_Player)
        # Get Q values for all actions
        q_values = NN_model.predict(currentState.reshape(1,18), batch_size=1)[0].reshape(3,3)
        # find best move
        _q_max_index, CurrentAction_index = find_max_qValue(q_values, stateX_Player, stateO_Player, -1.)

        # Apply action to state
        current_reward, stateX_Player, stateO_Player, terminal_state = apply_move(PlayerX_turn, stateX_Player,
                                                                                    stateO_Player, CurrentAction_index)

        if terminal_state: # end of game!
            if not current_reward:
                # Draw
                return move_num, False, False
            elif PlayerX_turn:
                # X wins
                return move_num, True, False
            # O wins
            return move_num, False, True

        PlayerX_turn = not PlayerX_turn
        move_num += 1


def find_equivalent_states(state, a):
    """
    find state/action equivalent transforms (rotations/flips).
    """
    # Get composite state and apply action to it (with reverse sign to distinct from existing marks)
    sa = np.sum(state, 0) - a

    # Transpose state from [channel, height, width] to [height, width, channel]
    state = np.transpose(state, [1, 2, 0])

    state_transposed = [state]
    action_transposed = [a]
    sa_transposed= [sa]

    # Apply rotations
    sa_next = sa
    for i in range(1, 4):  # rotate to 90, 180, 270 degrees
        sa_next = np.rot90(sa_next)
        if check_if_same_states(sa_transposed, sa_next):
            # Skip rotated state matching state already contained in list
            continue
        state_transposed.append(np.rot90(state, i))
        action_transposed.append(np.rot90(a, i))
        sa_transposed.append(sa_next)

    # Apply flips
    sa_next = np.fliplr(sa)
    if not check_if_same_states(sa_transposed, sa_next):
        state_transposed.append(np.fliplr(state))
        action_transposed.append(np.fliplr(a))
        sa_transposed.append(sa_next)
    sa_next = np.flipud(sa)
    if not check_if_same_states(sa_transposed, sa_next):
        state_transposed.append(np.flipud(state))
        action_transposed.append(np.flipud(a))
        sa_transposed.append(sa_next)

    return [np.transpose(s, [2, 0, 1]) for s in state_transposed], action_transposed


def check_if_same_states(s1, s2):
    """
    Check states s1 (or one of in case of array-like) and s2 are the same.
    """
    return np.any(np.isclose(np.mean(np.square(s1-s2), axis=(1, 2)), 0))


def get_state(PlayerX_turn, sx, so):
    """
    Create full state from X and O states.
    """
    return np.array([sx, so] if PlayerX_turn else [so, sx], dtype=np.float)


def find_max_qValue(q, sx, so, epsilon):
    """
    Choose action index for given state.
    """
    # Get valid action indices
    actionsValid_inds = np.where( (sx+so) == False)
    actionsTvalid_inds = np.transpose(actionsValid_inds)


    q_max_index = tuple(actionsTvalid_inds[np.argmax(q[actionsValid_inds])])

    # Choose next action based on epsilon-greedy policy
    if np.random.random() <= epsilon:
        # Choose random action from list of valid actions
        a_index = tuple(actionsTvalid_inds[np.random.randint(len(actionsTvalid_inds))])
    else:
        # Choose valid action w/ max Q
        a_index = q_max_index

    return q_max_index, a_index


def apply_move(PlayerX_turn, state_playerX, state_playerO, action_index):
    """
    Apply action to state, get reward and check for terminal state.
    """
    state = state_playerX if PlayerX_turn else state_playerO
    state[action_index] = True
    player_win_flag = check_board_wins(state)

    if player_win_flag:
        return WINNERS_REWARD, state_playerX, state_playerO, True
    if check_draw(state_playerX, state_playerO):
        return DRAW_REWARD, state_playerX, state_playerO, True
    return ACTION_REWARD, state_playerX, state_playerO, False


class TicTacToe:
    def __init__(self, playerX, playerO):
        self.board = [' ']*9
        self.playerX, self.playerO = playerX, playerO
        self.playerX_turn = random.choice([True, False])
        self.draws = 0


    def play_game(self):
        self.playerX.start_game('X')
        self.playerO.start_game('O')
        while True :
            if self.playerX_turn:
                player, char,other_char, other_player = self.playerX, 'X','O', self.playerO
            else:
                player, char,other_char, other_player = self.playerO, 'O','X', self.playerX
            # display board for human player
            if player.breed == "human":
                self.display_board()

            # read in the move, and move
            square = player.move(self.board,char)

            if square < 0:
                print ("Exiting game...")
                sys.exit(0)

            # correct index for humans and minmax (pick range of 1-9)
            if player.breed != 'Qplayer':
                square -= 1

            if self.board[square] != ' ':
                print ("bad move! stupid",player.breed,"!!!!!!!!!!!!!!!!!")
                print ("Ouch! something is there! try a new game")
                print ("current board:",self.board)
                print ("Last move:",square)
                print ("Restarting game...")
                break

            # place the char on board
            self.board[square] = char

            # check for wins/draws/losss
            if self.player_wins(char):
                if player.breed == 'human' or other_player.breed == 'human':
                    print ("Player", player.breed, "won the game!")
                break

            elif self.board_full() : # tie game
                if player.breed == 'human' or other_player.breed == 'human':
                    print("Tie game!")
                self.draws += 1
                break

            self.playerX_turn = not self.playerX_turn

    def player_wins(self, char):
        for a,b,c in [(0,1,2), (3,4,5), (6,7,8), #rows
                      (0,3,6), (1,4,7), (2,5,8), #columns
                      (0,4,8), (2,4,6)]:          #diagonals
            if char == self.board[a] == self.board[b] == self.board[c]:
                return True
        return False

    def board_full(self):
        return not any([square == ' ' for square in self.board])

    def display_board(self):
        row = " {} | {} | {}"
        hr = "\n-----------\n"
        print ((row + hr + row + hr + row).format(*self.board))


class Player(object):
    def __init__(self):
        self.breed = "human"
        self.gameCounter = 1
        self.moveCounter = 1
        self.wins = 0
        self.draws = 0

    def start_game(self, char):
        print ("\nNew game!")

    def move(self, board,char):
        while (True):
            try:
                inp = int(raw_input("Your move? [Enter a number between 1 to 9] "))
                if inp>=1 and inp<=9:
                    return inp
                elif inp < 0:
                    # "Exiting game"
                    return inp
                else:
                    print ("Ilegal move! try again\n")
            except:
                return 0

    def available_moves(self, board):
        return [i+1 for i in range(0,9) if board[i] == ' ']

class MinimaxPlayer(Player):
    def __init__(self):
        super(MinimaxPlayer, self).__init__()
        self.breed = "minimax"
        self.best_moves = {}

    def start_game(self, char):
        self.me = char
        self.enemy = self.other(char)

    def other(self, char):
        return 'O' if char == 'X' else 'X'

    def move(self, board, char):

        if tuple(board) in self.best_moves:
            return random.choice(self.best_moves[tuple(board)])
        if len(self.available_moves(board)) == 9:
            return random.choice([1, 3, 7, 9])
        best_yet = -2
        choices = []
        for move in self.available_moves(board):
            board[move - 1] = self.me
            optimal = self.minimax(board, self.enemy, -2, 2)
            board[move - 1] = ' '
            if optimal > best_yet:
                choices = [move]
                best_yet = optimal
            elif optimal == best_yet:
                choices.append(move)
        self.best_moves[tuple(board)] = choices
        return random.choice(choices)

    def minimax(self, board, char, alpha, beta):
        if self.player_wins(self.me, board):
            return 1
        if self.player_wins(self.enemy, board):
            return -1
        if self.board_full(board):
            return 0
        for move in self.available_moves(board):
            board[move - 1] = char
            val = self.minimax(board, self.other(char), alpha, beta)
            board[move - 1] = ' '
            if char == self.me:
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    return beta
            else:
                if val < beta:
                    beta = val
                if beta <= alpha:
                    return alpha
        if char == self.me:
            return alpha
        else:
            return beta

    def player_wins(self, char, board):
        for a, b, c in [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                        (0, 3, 6), (1, 4, 7), (2, 5, 8),
                        (0, 4, 8), (2, 4, 6)]:
            if char == board[a] == board[b] == board[c]:
                return True
        return False

    def board_full(self, board):
        return not any([square == ' ' for square in board])


class QLearningPlayer(Player):
    def __init__(self,NN_model):
        super(QLearningPlayer, self).__init__()
        self.DQNN = NN_model
        self.breed = "Qplayer"


    def start_game(self, char):
        self.last_board = (' ',) * 9
        self.last_move = None

    def getQValues(self, state):
        predicted = self.DQNN.predict(state.reshape(1, 18), batch_size=1)
        return predicted[0]


    def move(self,board,char):
        PlayerX_turn = True if char =='X' else False
        stateX_Player = np.asarray([True if charI =='X' else False for charI in board ]).reshape(size_board,size_board)
        stateO_Player = np.asarray([True if charI =='O' else False for charI in board ]).reshape(size_board,size_board)

        currentState= get_state(PlayerX_turn, stateX_Player, stateO_Player)
        q_values = self.DQNN.predict(currentState.reshape(1, 18), batch_size=1)[0].reshape(3, 3)
        _q_max_index, CurrentAction_index = find_max_qValue(q_values, stateX_Player, stateO_Player, -1.) # pay attention to epsilon!
        return CurrentAction_index[0]*size_board + CurrentAction_index[1]


if __name__ == "__main__":

    # input values for NN
    inputs = 18
    outputs = 9
    hidden_layer_nodes = [150]
    leraning_rate = 0.001

    #if sys.flags.interactive:
    if sys.stdin.isatty():
        if len(sys.argv) < 2 :
            print ("Loading model from current folder...")
            # load model and init Q_player
            try:
                NN_model_loaded = load_model('NN_TTT_model.h5')
                pQNN = QLearningPlayer(NN_model_loaded)
            except:
                print ("Model isn't in current directory")

        elif sys.argv[1] == 'train':
            # train network from scratch
            NN_model = create_neuralNetwork(inputs, outputs, hidden_layer_nodes, leraning_rate)
            print ("Interactive mode: Training a neural network...")

            train(NN_model)
            pQNN = QLearningPlayer(NN_model)
        else:
            print ("Usage: python TTT_RL.py [train]")
            sys.exit(0)
    else:
        NN_model = create_neuralNetwork(inputs, outputs, hidden_layer_nodes, leraning_rate)
        print("Training a neural network...")
        train(NN_model)
        # save to JSON
        model_json = NN_model.to_json()
        print ("saving model to Json")
        with open("NN_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        NN_model.save_weights("NN_TTT_model_2.h5")
        pQNN = QLearningPlayer(NN_model)

    # NN_model_loaded = load_model('NN_TTT_model.h5')
    # pQNN = QLearningPlayer(NN_model_loaded)
    print ("Lets play!")
    pHuman = Player()
    while(True):
        t = TicTacToe(pQNN, pHuman)
        t.play_game()




