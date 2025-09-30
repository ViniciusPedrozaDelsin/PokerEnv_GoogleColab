from pokerEnv import Game, Player, Dealer, Table
from collections import Counter

DEALER = Dealer()
TABLE = Table()
PLAYER_ZERO = Player("Vinicius", model_name="model_1", update_model=True)
PLAYER_ONE = Player("Player 1")
'''PLAYER_TWO = Player("Player 2")
PLAYER_THREE = Player("Player 3")
PLAYER_FOUR = Player("Player 4")
PLAYER_FIVE = Player("Player 5")
PLAYER_SIX = Player("Player 6")
PLAYER_SEVEN = Player("Player 7")
PLAYER_EIGHT = Player("Player 8")
PLAYER_NINE = Player("Player 9")'''

winners = []
i = 1
while i < 30:
	PokerGame = Game()
	PokerGame.add_table(TABLE)
	PokerGame.add_dealer(DEALER)
	PokerGame.add_player(PLAYER_ZERO)
	PokerGame.add_player(PLAYER_ONE)
	'''PokerGame.add_player(PLAYER_TWO)
	PokerGame.add_player(PLAYER_THREE)
	PokerGame.add_player(PLAYER_FOUR)
	PokerGame.add_player(PLAYER_FIVE)
	PokerGame.add_player(PLAYER_SIX)
	PokerGame.add_player(PLAYER_SEVEN)
	PokerGame.add_player(PLAYER_EIGHT)
	PokerGame.add_player(PLAYER_NINE)'''
	PokerGame.start()
	winners.append(PokerGame.winner)
	print(f"End Game {i}")
	i += 1

PLAYER_ZERO.saveModels()

counts = Counter(winners)
print(counts)
