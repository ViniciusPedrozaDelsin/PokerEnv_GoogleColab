class Table:
    def __init__(self, n_of_positions=10):
        self.number_of_positions = n_of_positions
        self.table_positions = list(range(0, self.number_of_positions))
        self.available_positions = self.table_positions
        self.cards = []
        self.cards_encoded = []
        self.show_cards = False
        self.pot = 0
        self.pot_round = 0
        self.pot_round_list = []
    
    # Print Output
    def __str__(self):
        return f"Table Positions: {self.number_of_positions} | Available Positions {self.available_positions} | Pot Size: {self.pot} | Cards {self.cards_encoded}"
    
    # Add Community Card
    def add_card(self, card):
        self.cards.append(card)
    
    # Add Community Card Encoded
    def add_card_encoded(self, card):
        self.cards_encoded.append(card)
    
    # Add Bet on the Pot
    def add_bet_on_pot(self, bet):
        self.pot += bet
        
    # Hard Reset Table
    def hardResetTable(self):
        self.table_positions = list(range(0, self.number_of_positions))
        self.available_positions = self.table_positions
        self.cards = []
        self.cards_encoded = []
        self.show_cards = False
        self.pot = 0
        self.pot_round = 0
        self.pot_round_list = []
    
    # Soft Reset Table
    def softResetTable(self):
        self.cards = []
        self.cards_encoded = []
        self.show_cards = False
        self.pot = 0
        self.pot_round = 0
        self.pot_round_list = []
    
    # Set Show Cards
    def set_show_cards(self, boolean):
        self.show_cards = boolean
    
    # Increase Pot
    def increase_pot(self, value):
        self.pot += value
    
    # Decrease Pot
    def decrease_pot(self, value):
        self.pot -= value
    
    # Fill Position
    def fillPosition(self):
        position = self.available_positions[0]
        self.available_positions.remove(self.available_positions[0])
        return position