"""
Implementation of Valued Missing Values Identifier engine class to run valued missing value analysis.
"""

from typing import Optional

import pandas as pd

from ydata_quality.core import QualityEngine, QualityWarning


class VMVIdentifier(QualityEngine):
    "Engine for running analysis on valued missing values."

    def __init__(self, df: pd.DataFrame, vmv_extensions: Optional[list]=[], random_state=42):
        """
        Args:
            df (pd.DataFrame): DataFrame used to run the missing value analysis.
            vmv_extensions: A list of user provided Value Missing Values to append to defaults.
        """
        super().__init__(df=df, random_state=random_state)
        self._tests = ["flatlines", "predefined_valued_missing_values"]
        self._default_vmvs = None
        self._flatline_index = {}
        self.__default_index_name = '__index'
        self.vmvs = vmv_extensions

    @property
    def default_vmvs(self):
        """Returns the default list of Valued Missing Values.
        VMVs of string type are case insensitive during search."""
        if self._default_vmvs is None:
            self._default_vmvs = set([vmv.lower() if isinstance(vmv, str) else vmv for vmv in ["?", "UNK", "Unknown", "N/A", "NA", "", "(blank)"]])
        return self._default_vmvs

    @property
    def vmvs(self):
        """Returns the extended Value Missing Values (default plus user provided).
        VMVs of string type are case insensitive during search."""
        if not  self._vmvs:
            self._vmvs = self.default_vmvs
        return self._vmvs

    @vmvs.setter
    def vmvs(self, vmv_extensions: Optional[list] = []):
        """Allows extending default Valued Missing Values list, append only.
        VMVs of string type are case insensitive during search."""
        assert isinstance(vmv_extensions, list), "vmv extensions must be passed as a list"
        self._vmvs = self.default_vmvs.union(set([vmv.lower() if isinstance(vmv, str) else vmv for vmv in vmv_extensions]))

    def __get_flatline_index(self, column_name: str, th: Optional[int] = 1):
        """Returns an index for flatline events on a passed column.
        A flatline event is any full sequence of repeated values on the column.
        The returned index is a compact representation of all occurrences of flatline events.
        Returns a DataFrame with index equal to the index of first element in the event,
        a tail column identifying the last element of the sequence and a length column."""
        if column_name in self._flatline_index:  # Read from index cache
            flts = self._flatline_index[column_name]
        else:  # Produce and cache index
            df = self.df.copy()  # Index will not be covered in column iteration
            if column_name == self.__default_index_name:
                df[self.__default_index_name] = df.index  # Index now in columns to be processed next
            column  = df[column_name]
            column.fillna('__filled')  # So NaN values are considered
            sequence_indexes = column.ne(column.shift()).cumsum()  # Everytime shifted value is different from previous a new sequence starts
            sequence_groups = column.index.to_series().groupby(sequence_indexes)  # Group series indexes by sequence indexes
            data = {'length': sequence_groups.count().values,
            'ends': sequence_groups.last().values}
            flts = pd.DataFrame(data, index=sequence_groups.first().values).query('length > 1')  # Just dropping single unique values (detected as independent sequences)
            flts.rename_axis('starts', inplace=True)  # Adding index name for clarity
            self._flatline_index[column_name] = flts  # Cache the index
        return flts.loc[flts['length']>=th]

    def flatlines(self, th: int=5, skip: list=[]):
        """Iterates the dataset over columns and requests flatline indexes based on arguments.
        Raises warning indicating columns with flatline events and total flatline events in the dataframe.
        Arguments:
            th: Defines the minimum length required for a flatline event to be reported.
            skip: List of columns that will not be target of search for flatlines.
                Pass '__index' inside skip list to skip looking for flatlines at the index."""
        flatlines = {}
        for column in self.df.columns:  # Compile flatline index
            if column in skip:
                continue  # Column not requested
            flts = self.__get_flatline_index(column, th)
            if len(flts) > 0:
                flatlines[column] = flts
        if len(flatlines)>0:  # Flatlines detected
            total_flatlines = sum([flts.shape[0] for flts in flatlines.values()])
            self.store_warning(
                QualityWarning(
                    test='Flatlines', category='Valued Missing Values', priority=2, data=flatlines,
                    description=f"Found {total_flatlines} flatline events with a minimun length of {th} among the columns {set(flatlines.keys())}."
            ))
            return flatlines
        else:
            print("[FLATLINES] No flatline events with a minimum length of {} were found.".format(th))

    def predefined_valued_missing_values(self, skip: list=[], short: bool = True):
        """Runs a check against a list of predefined Valued Missing Values.
        Will always use the extended list if user provided any.
        Raises warning based on the existence of these values.
        VMVs of string type are case insensitive during search.
        Returns a DataFrame with count distribution for each predefined type over each column.
        Arguments:
            skip: List of columns that will not be target of search for vmvs.
                Pass '__index' in skip to skip looking for flatlines at the index.
            short: Instruct engine to return only for VMVs and columns where VMVs were detected"""
        df = self.df.copy()  # Index will not be covered in column iteration
        df[self.__default_index_name] = df.index  # Index now in columns to be processed
        check_cols = set(df.columns).difference(set(skip))
        df = df[check_cols]
        vmvs = pd.DataFrame(index=self._vmvs, columns=check_cols)
        for vmv in self._vmvs:
            check_vmv = lambda x: x.lower()==vmv if isinstance(x,str) else x==vmv
            vmvs.loc[vmv] = df.applymap(check_vmv).sum()
        if short:
            no_vmv_cols = vmvs.columns[vmvs.sum()==0]
            no_vmv_rows = vmvs.index[vmvs.sum(axis=1)==0]
            vmvs.drop(no_vmv_cols, axis=1, inplace=True)
            vmvs.drop(no_vmv_rows, inplace=True)
        if vmvs.empty:
            print("[PREDEFINED VALUED MISSING VALUES] No predefined vmvs from  the set {} were found in the dataset.".format(
                self.vmvs
            ))
        else:
            total_vmvs = vmvs.sum().sum()
            self.store_warning(
                QualityWarning(
                    test='Predefined Valued Missing Values', category='Valued Missing Values', priority=2, data=vmvs,
                    description=f"Found {total_vmvs} vmvs in the dataset."
            ))
            return vmvs
