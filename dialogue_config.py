# Special slot values (for reference)
'PLACEHOLDER'  # For informs
'UNK'  # For requests
'anything'  # means any value works for the slot with this value
'no match available'  # When the intent of the agent is match_found yet no db match fits current constraints

# All slot to be defined
unsupport_slots = ['name_customer', 'addr_customer', 'shipping_fee',
    'height_customer', 'weight_customer', 'phone_customer',
    'bust_customer', 'waist_customer', 'hip_customer',
    'delivery_time']

size_slots = [# "size_customer", 
    # "bust_customer", 
    "waist_customer", 
    # "hip_customer", 
    "height_customer", "weight_customer"
]

#######################################
# Usersim Config
#######################################
# Used in EMC for intent error (and in user)
usersim_intents = ['inform', 'request', 'ok', 'reject', 'done']
# usersim_intents = ['inform', 'request', 'done']

# The goal of the agent is to inform a match for this key
usersim_default_key = 'shopping'

# Required to be in the first action in inform slots of the usersim if they exist in the goal inform slots
usersim_required_init_inform_keys = ['name_product']

#######################################
# Agent Config
#######################################

# Possible inform and request slots for the agent
agent_inform_slots = ['name_product', 'size_product', 'color_product', 'material_product', 'cost_product', 
                    # 'amount_product', 
                    ] + [usersim_default_key]
# agent_inform_slots = ['cost_product']
agent_request_slots = ['name_product', 'size_product', 'color_product', 
                    'material_product', 'cost_product', 'amount_product'] + size_slots

# Possible actions for agent
agent_actions = [
    {'intent': 'done', 'inform_slots': {}, 'request_slots': {}},  # Triggers closing of conversation
    {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
]
for slot in agent_inform_slots:
    # Must use intent match found to inform this, but still have to keep in agent inform slots
    if slot == usersim_default_key:
        continue
    agent_actions.append({'intent': 'inform', 'inform_slots': {slot: 'PLACEHOLDER'}, 'request_slots': {}})
for slot in agent_request_slots:
    agent_actions.append({'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}})

# Rule-based policy request list
rule_requests = ['name_product', 'size_product'] + size_slots + ['color_product', 
                'material_product', 'cost_product', 'amount_product']

# These are possible inform slot keys that cannot be used to query
no_query_keys = ['amount_product', usersim_default_key]

#######################################
# Global config
#######################################

# These are used for both constraint check AND success check in usersim
FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1
# UNSUITABLE = -2
# GOOD_INFORM = 2
# NO_VALUE = 3

# All possible intents (for one-hot conversion in ST.get_state())
# all_intents = ['inform', 'request', 'done']
all_intents = ['inform', 'request', 'done', 'match_found', 'ok', 'reject']

# All possible slots (for one-hot conversion in ST.get_state())
all_slots = ['name_product', 'size_product', 'color_product', 
'cost_product', 'material_product', 'amount_product'] + size_slots + [usersim_default_key]

# All constraints slots for request product info goal
request_product_entity = {
    'all_keys': ['name_product', 'size_product', 'color_product', 'cost_product', 'material_product', 'amount_product'],
    'name_product': ['size_product', 'color_product', 'cost_product', 'material_product', 'amount_product'],
    'material_product': ['name_product'],
    'size_product': ['name_product', 'color_product'],
    'color_product': ['name_product', 'size_product'],
    'cost_product': ['name_product'],
    'amount_product': ['name_product', 'size_product', 'color_product']
}
