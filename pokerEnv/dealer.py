from .deck import Deck

class Dealer(Deck):
    def __init__(self):
        super().__init__()
        self.table = None
        self.players = []
        self.n_of_players = 0
        self.n_of_players_round = 0
        self.minimum_bet = None
        self.maximum_bet = None
        self.round_final_players_order = []
    
    # Reset Dealer
    def resetDealer(self):
        self.table = None
        self.players = []
        self.n_of_players = 0
        self.n_of_players_round = 0
        self.resetDeck()
        self.minimum_bet = None
        self.maximum_bet = None
        self.dict_round_position = {}
        self.round_final_players_order = []
    
    # Add a Player
    def addPlayer(self, player):
        self.players.append(player)
    
    # Add the Table
    def addTable(self, table):
        self.table = table
    
    # Set Minimum Bet
    def set_minimum_bet(self, value):
        self.minimum_bet = value
    
    # Set Minimum Bet
    def set_maximum_bet(self, value):
        self.maximum_bet = value
    
    # Cut the Deck
    def cutDeck(self):
        pass
    
    # Deal Cards
    def deal(self):
        for player in self.players:
            first_card = self.get_top_card()
            second_card = self.get_top_card()
            player.set_hand(first_card, second_card)
            player.set_hand_encoded(self.encodingCards(first_card), self.encodingCards(second_card))
    
    # Burn Card
    def burnCard(self):
        self.get_top_card()
    
    # Open Flop
    def openFlop(self):
        flop_first_card = self.get_top_card()
        self.table.add_card(flop_first_card)
        self.table.add_card_encoded(self.encodingCards(flop_first_card))
        
        flop_second_card = self.get_top_card()
        self.table.add_card(flop_second_card)
        self.table.add_card_encoded(self.encodingCards(flop_second_card))
        
        flop_third_card = self.get_top_card()
        self.table.add_card(flop_third_card)
        self.table.add_card_encoded(self.encodingCards(flop_third_card))
    
    # Open Turn
    def openTurn(self):
        turn_card = self.get_top_card()
        self.table.add_card(turn_card)
        self.table.add_card_encoded(self.encodingCards(turn_card))
    
    # Open River
    def openRiver(self):
        river_card = self.get_top_card()
        self.table.add_card(river_card)
        self.table.add_card_encoded(self.encodingCards(river_card))
    
    # Remove Player from Dict Round Positions
    def removePlayerRoundPosition(self, player_name):
        self.dict_round_position = self.dict_round_position.pop(player_name)
    
    # Set Round Positions
    def set_game_positions(self):
        self.dict_round_position = {}
        i = 0
        #n_of_positions = self.table.number_of_positions
        n_of_positions = len(self.players)
        for player in self.players:
            player_round_position = player.set_round_position(i, n_of_positions)
            self.dict_round_position[player_round_position[0]] = [player_round_position[1], player_round_position[2]]
            i += 1
    
    # Rotate Round Position
    def rotate_game_positions(self):
        self.dict_round_position = {}
        n_of_positions = len(self.players)
        players_sorted = sorted(self.players, key=lambda player: player.round_position)
        layers_sorted = [players_sorted[-1]] + players_sorted[:-1]
        i = 0
        for player in layers_sorted:
            player_round_position = player.set_round_position(i, n_of_positions)
            self.dict_round_position[player_round_position[0]] = [player_round_position[1], player_round_position[2]]
            i += 1
    
    # Set Number of Players
    def set_number_of_player(self):
        self.n_of_players = len(self.players)
    
    # Place Bet of a Player
    def place_bet(self, player):
        pass