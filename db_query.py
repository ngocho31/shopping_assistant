from collections import defaultdict
import copy

from utils import DEBUG_PRINT

class DBQuery:
    """Queries the database for the state tracker."""

    def __init__(self, database):
        """
        The constructor for DBQuery.

        Parameters:
            database (dict): The database in the format dict(long: dict)
        """

        self.database = database
        # {frozenset: {string: int}} A dict of dicts
        self.cached_db_slot = defaultdict(dict)
        # {frozenset: {'#': {'slot': 'value'}}} A dict of dicts of dicts, a dict of DB sub-dicts
        self.cached_db = defaultdict(dict)
