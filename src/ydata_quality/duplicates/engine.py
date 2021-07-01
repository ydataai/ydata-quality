"""
Implementation of DuplicateChecker engine class to run duplicate records analysis.
"""

from typing import List, Optional, Union

import pandas as pd
from ydata_quality.core import QualityWarning, QualityEngine

class DuplicateChecker(QualityEngine):
    "Engine for running analyis on duplicate records."

    def __init__(self, df: pd.DataFrame, entities: List[Union[str, List[str]]] = []):
        self._df = df
        self._entities = entities
        self._warnings = set()
        self._tests = ["exact_duplicates", "entity_duplicates", "duplicate_columns"]

    @property
    def entities(self):
        "Property that returns the entities relevant for duplicates analysis."
        return self._entities

    @entities.setter
    def entities(self, entities: List[Union[str, List[str]]]):
        if not isinstance(entities, list):
            raise ValueError("Property 'entities' should be a list.")
        entities = self.__unique_entities(entities)
        assert all(entity in self.df.columns if isinstance(entity, str) else [c in self.df.columns for c in entity] for entity in entities), "Given entities should exist as DataFrame's columns."
        self._entities = entities

    @staticmethod
    def __unique_entities(entities: List[Union[str, List[str]]]):
        """Returns entities list with only unique entities"""
        entities = set([entity if isinstance(entity, str) else entity[0] if len(entity)==1 else tuple(entity) for entity in entities])
        return [entity if isinstance(entity, str) else list(entity) for entity in entities]

    @staticmethod
    def __get_duplicates(df: pd.DataFrame):
        "Returns duplicate records."
        return df[df.duplicated()]

    @staticmethod
    def __get_entity_duplicates(df: pd.DataFrame, entity: Union[str, List[str]]):
        "Returns the duplicate records aggregated by a given entity."
        return df.groupby(entity).apply(DuplicateChecker.__get_duplicates).reset_index(drop=True)

    def exact_duplicates(self):
        "Returns a DataFrame filtered for exact duplicate records."
        dups = self.__get_duplicates(self.df) # Filter for duplicate instances
        if len(dups) > 0:
            self._warnings.add(
                QualityWarning(
                    test='Exact Duplicates', category='Duplicates', priority=2, data=dups,
                    description=f"Found {len(dups)} instances with exact duplicate feature values."
            ))
        else:
            print("[EXACT DUPLICATES] No exact duplicates were found.")
            dups = None
        return dups

    def entity_duplicates(self, entity: Optional[Union[str, List[str]]] = None):
        """Returns a dict of {entity: {entity_value: duplicates}} of duplicate records after grouping by an entity.
        If entity is not specified, compute for all entities defined in the init.
        """
        # Computation Decision Tree
        # entity is specified : compute for given entity, return dict {entity_values: duplicate records}
        # entity is not specified
        #   -> entities is None : skip the test
        #   -> entities is not None : defaults to entities defined in the init
        ent_dups = {}
        if entity is not None: # entity is specified
            dups = self.__get_entity_duplicates(self.df, entity)
            if len(dups) > 0:                        # if we have any duplicates
                self._warnings.add(
                    QualityWarning(
                        test='Entity Duplicates', category='Duplicates', priority=2, data=dups,
                        description=f"Found {len(dups)} duplicates after grouping by entities."
                ))
                if isinstance(entity, str):
                    entity = [entity]  # Makes logic the same for str or List[str] entities
                set_vals = set(dups[entity].apply(tuple, axis=1))
                if len(entity)>1:
                    entity_key = tuple(entity)  # Lists are not hashable, therefore cannot be dictionary keys
                else:
                    set_vals = [val[0] for val in set_vals]  # No need to store keys as tuples for single entities (single values)
                    entity_key = entity[0]
                for val in set_vals:  # iterate on each entity with duplicates
                    ent_dups.setdefault(entity_key, {})[val] = dups[(dups[entity].values==val).all(axis=1)]
        else: # if entity is not specified
            if len(self.entities) == 0:
                print("[ENTITY DUPLICATES] There are no entities defined to run the analysis. Skipping the test.")
                return None
            else:
                for col in self.entities:
                    ent_dups.update(self.entity_duplicates(col))
        return ent_dups


    def duplicate_columns(self):
        "Returns a mapping dictionary of columns with fully duplicated feature values."
        dups = {}
        for idx, col in enumerate(self.df.columns): # Iterate through all the columns of dataframe
            ref = self.df[col]                      # Take the column values as reference.
            for tgt_col in self.df.columns[idx+1:]: # Iterate through all other columns
                if ref.equals(self.df[tgt_col]):    # Take target values
                    dups[col] = tgt_col  # Store if they match

        if len(dups) > 0:
            self._warnings.add(
                QualityWarning(
                    test='Duplicate Columns', category='Duplicates', priority=1, data=dups,
                    description=f"Found {len(dups)} columns with exactly the same feature values as other columns."
                )
            )
        else:
            print("[DUPLICATE COLUMNS] No duplicate columns were found.")
            dups = None
        return dups