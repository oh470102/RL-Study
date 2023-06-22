import numpy as np

class GridWorld:
    def __init__(self, size, mode):
        self.size = size
        self.mode = mode
        self.board = self.setup_board([[0 for i in range(self.size)] for j in range(self.size)])
        self.reward = None
    
    def setup_board(self, board):

        # +, -, W, P
        choose_from = [i for i in range(0, self.size**2)]
        idxs = np.random.choice(choose_from, size=4, replace=False)
        idxs = [(idx%self.size, idx//self.size) for idx in idxs]

        items_choose_from = ['+', '-', 'W', 'P']
        np.random.shuffle(items_choose_from)

        for row in range(0, self.size):
            for col in range(0, self.size):
                if (row, col) in idxs:
                    board[row][col] = items_choose_from[0]
                    del items_choose_from[0]
                else:
                    board[row][col] = ' '

        return np.array(board)

    def display(self):
        print((self.board))

    def makeMove(self, move):
        # u, d, l, r

        pos_axis1, pos_axis0 = int(np.where(self.board == 'P')[0].item()), int(np.where(self.board == 'P')[1].item())
        orig_pos_axis1, orig_pos_axis0 = pos_axis1, pos_axis0

        if move == 'u':
            pos_axis1 -= 1
            
        elif move == 'd':
            pos_axis1 += 1

        elif move == 'l':
            pos_axis0 -= 1

        elif move == 'r':
            pos_axis0 += 1

        # INVALID MOVE FILTERING
        if pos_axis0 < 0 or pos_axis0 > self.size-1:
            self.reward = -10
            return "INVALID MOVE: OUT OF BOUND"
        if pos_axis1 < 0 or pos_axis1 > self.size-1: 
            self.reward = -10
            return "INVALID MOVE: OUT OF BOUND"
        if self.board[pos_axis1, pos_axis0] == 'W':  
            self.reward = -1
            return "INVALID MOVE: WALL"
       
        # IF MOVE IS VALID:
        self.board[orig_pos_axis1, orig_pos_axis0] = ' ' # 원래 위치는 공백으로
        landed_at = self.board[pos_axis1, pos_axis0] # 도착 지점에 있던 것
        self.board[pos_axis1, pos_axis0] = 'P' # 플레이어 위치 변경

        # 도착한 곳이 종료점(+) 또는 함정(-) 인 경우
        if landed_at == '-':  
            self.reward = -10
            return "GAME OVER: TRAP"
        elif landed_at == '+':
            self.reward = 10
            return "GAME OVER: WON"
        else:
            self.reward = -1

    def give_reward(self):
        return self.reward

    def render_np(self):
        board_np = np.zeros((4,4,4))

        #1 and 0 indicate axis number
        player1, player0 = np.where(self.board=='P')[0].item(), np.where(self.board=='P')[1].item()
        wall1, wall0 = np.where(self.board=='W')[0].item(), np.where(self.board=='W')[1].item()
        plus1, plus0 = np.where(self.board=='+')[0].item(), np.where(self.board=='+')[1].item()
        minus1, minus0 = np.where(self.board=='-')[0].item(), np.where(self.board=='-')[1].item()

        # REMEMBER: (axis 2, axis 1, axis 0)
        # 앞에서부터, (플레이어, 목표, 구덩이, 벽)
        board_np[0, player1, player0] = 1
        board_np[3, wall1, wall0] = 1
        board_np[1, plus1, plus0] = 1
        board_np[2, minus1, minus0] = 1

        return board_np

