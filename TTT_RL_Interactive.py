# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import random
from keras.models import load_model
from keras.models import model_from_json

import sys
from PyQt4 import QtGui

# Board size
size_board = 3

class IATicTacToe(QtGui.QWidget):
    def __init__(self,NN_model):
        QtGui.QWidget.__init__(self)

        self.setMinimumSize(300, 300)
        QtGui.QWidget.setWindowTitle(self,"Tic-Tac-Toe Forever")
        self.fieldLayouts = []
        self.fields = []
        self.layout = QtGui.QVBoxLayout()
        self.Currplayer = random.choice(['human','Qplayer'])

        self.DQNN = NN_model


        ###########

        # Create nine buttons
        for x in range(9):
            button = QtGui.QPushButton("")
            #button.resize(20,20)
            self.fields.append(button)

        # Connect the buttons
        for x in self.fields:
            x.clicked.connect(self.getMove)

        # Create three QHBoxLayouts for buttons
        for x in range(3):
            row = QtGui.QHBoxLayout()
            self.fieldLayouts.append(row)

        # Fill the layouts (3x3)
        i = 0
        for x in self.fieldLayouts:
            for y in range(3):
                x.addWidget(self.fields[i])
                #x.setGeometry()
                i = i + 1

        # Add the layouts to the main layout
        for x in self.fieldLayouts:
            self.layout.addLayout(x)

        # Set the main layout
        self.setLayout(self.layout)

        # start the game ?
        self.QplayerStart()



    def getMove(self):

        # human player moves
        button = self.sender()
        button.setText('X')
        board = [str(x.text()) for x in self.fields]
        terminal = self.checkTerminalState(board)
        if not terminal:
            # Qplayer moves
            action = self.moveQ(board, 'O')
            self.fields[action].setText('O')

            # check for wins or draws
            Lastboard = [str(x.text()) for x in self.fields]
            self.checkTerminalState(Lastboard)
           # print (Lastboard)


    def checkTerminalState(self,board):
        gameOver, char = self.player_wins(board)
        Winningplayer = 'Human' if char == 'X' else 'God'
        msgBox = QtGui.QMessageBox()
        if gameOver:
            msgBox.setText("We have a winner!\n" + Winningplayer + " has won!")
            msgBox.exec_()
            # Reset fields
            for x in self.fields:
                x.setText('')
            self.QplayerStart()
            return True


        # draws
        if self.board_full(board):
            msgBox.setText("Draw game\nWell done!")
            msgBox.exec_()            # Reset fields
            for x in self.fields:
                x.setText('')
            self.QplayerStart()
            return True
        return False


    def QplayerStart(self):
    # start a new game with prob 1/2
        if random.random() < 0.5:
            board = [str(x.text()) for x in self.fields]
            # Qplayer moves
            action = self.moveQ(board, 'O')
            self.fields[action].setText('O')


    def get_state(self,PlayerX_turn, sx, so):
        """
        Create full state from X and O states.
        """
        return np.array([sx, so] if PlayerX_turn else [so, sx], dtype=np.float)

    def find_max_qValue(self,q, sx, so, epsilon):
        """
        Choose action index for given state.
        """
        # Get valid action indices
        actionsValid_inds = np.where((sx + so) == False)
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

    def moveQ(self,board,char):
        PlayerX_turn = True if char =='X' else False
        stateX_Player = np.asarray([True if charI =='X' else False for charI in board ]).reshape(size_board,size_board)
        stateO_Player = np.asarray([True if charI =='O' else False for charI in board ]).reshape(size_board,size_board)

        currentState= self.get_state(PlayerX_turn, stateX_Player, stateO_Player)
        q_values = self.DQNN.predict(currentState.reshape(1, 18), batch_size=1)[0].reshape(3, 3)
        _q_max_index, CurrentAction_index = self.find_max_qValue(q_values, stateX_Player, stateO_Player, -1.) # pay attention to epsilon!
        return CurrentAction_index[0]*size_board + CurrentAction_index[1]

    def player_wins(self, board):
        for a,b,c in [(0,1,2), (3,4,5), (6,7,8), #rows
                      (0,3,6), (1,4,7), (2,5,8), #columns
                      (0,4,8), (2,4,6)]:          #diagonals
            if  board[a] == board[b] == board[c] !='':
                return True,board[a]
        return False,[]

    def board_full(self,board):
        return not any([square == '' for square in board])


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    NN_model_loaded = load_model('NN_TTT_model.h5')
    t = IATicTacToe(NN_model_loaded)
    t.show()

    sys.exit(app.exec_())





