import random

from utils import DEBUG_PRINT, SAVE_LOG
from dialogue_config import usersim_intents, size_slots


class ErrorModelController:
    """Adds error to the user action."""

    def __init__(self, db_dict, size_db_dict, constants):
        """
        The constructor for ErrorModelController.

        Saves items in constants, etc.

        Parameters:
            db_dict (dict): The database dict with format dict(string: list) where each key is the slot name and
                            the list is of possible values
            constants (dict): Loaded constants in dict
        """

        # print("caller ErrorModelController __init__")
        self.shopping_dict = db_dict
        self.size_shopping_dict = size_db_dict
        self.slot_error_prob = constants['emc']['slot_error_prob']
        self.slot_error_mode = constants['emc']['slot_error_mode']  # [0, 3]
        self.intent_error_prob = constants['emc']['intent_error_prob']
        self.intents = usersim_intents
        self.size_slots = size_slots

    def infuse_error(self, frame):
        """
        Takes a semantic frame/action as a dict and adds 'error'.

        Given a dict/frame it adds error based on specifications in constants. It can either replace slot values,
        replace slot and its values, delete a slot or do all three. It can also randomize the intent.

        Parameters:
            frame (dict): format dict('intent': '', 'inform_slots': {}, 'request_slots': {}, 'round': int,
                          'speaker': 'User')
        """

        # print("caller ErrorModelController infuse_error")
        informs_dict = frame['inform_slots']
        for key in list(frame['inform_slots'].keys()):
            assert key in self.shopping_dict or key in self.size_shopping_dict
            if random.random() < self.slot_error_prob:
                if self.slot_error_mode == 0:  # replace the slot_value only
                    self._slot_value_noise(key, informs_dict)
                elif self.slot_error_mode == 1:  # replace slot and its values
                    self._slot_noise(key, informs_dict)
                elif self.slot_error_mode == 2:  # delete the slot
                    self._slot_remove(key, informs_dict)
                else:  # Combine all three
                    rand_choice = random.random()
                    if rand_choice <= 0.33:
                        self._slot_value_noise(key, informs_dict)
                    elif rand_choice > 0.33 and rand_choice <= 0.66:
                        self._slot_noise(key, informs_dict)
                    else:
                        self._slot_remove(key, informs_dict)
                # DEBUG_PRINT("user (error) informs_dict:\t", informs_dict)
        if random.random() < self.intent_error_prob:  # add noise for intent level
            frame['intent'] = random.choice(self.intents)
            # DEBUG_PRINT("user (error) intent:\t", frame['intent'])
            # print("infuse_error intent = ", frame['intent'])

    def _slot_value_noise(self, key, informs_dict):
        """
        Selects a new value for the slot given a key and the dict to change.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        # print("caller ErrorModelController _slot_value_noise")
        if key in size_slots:
            noise_value = random.choice(self.size_shopping_dict[key])
        else:
            noise_value = random.choice(self.shopping_dict[key])

        informs_dict[key] = noise_value
        # val = random.choice(self.shopping_dict[key])
        # if key == 'amount_product':
        #     informs_dict.update({key: val})
        # else:
        #     informs_dict.update({key: [val]})

    def _slot_noise(self, key, informs_dict):
        """
        Replaces current slot given a key in the informs dict with a new slot and selects a random value for this new slot.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        # print("caller ErrorModelController _slot_noise")
        informs_dict.pop(key)
        if key in size_slots:
            random_slot = random.choice(list(self.size_shopping_dict.keys()))
            informs_dict[random_slot] = random.choice(self.size_shopping_dict[random_slot])
        else:
            random_slot = random.choice(list(self.shopping_dict.keys()))
            informs_dict[random_slot] = random.choice(self.shopping_dict[random_slot])
        # val = random.choice(self.shopping_dict[random_slot])
        # informs_dict[random_slot] = [val]

    def _slot_remove(self, key, informs_dict):
        """
        Removes the slot given the key from the informs dict.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        # print("caller ErrorModelController _slot_remove")
        informs_dict.pop(key)
