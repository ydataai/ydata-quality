"""
Implementation of Valued Missing Values Identifier engine class to run valued missing value analysis.
"""

from typing import Optional

import pandas as pd

from ydata_quality.core import QualityEngine, QualityWarning


class VMVIdentifier(QualityEngine):
    "Engine for running analysis on valued missing values."

    def __init__(self, df: pd.DataFrame, vmv_extensions: Optional[list]=[]):
        """
        Args:
            df (pd.DataFrame): DataFrame used to run the missing value analysis.
        """
        super().__init__(df=df)
        self._tests = ["flatlines", "predefined_valued_missing_values"]
        self._default_vmvs = None
        self._flatline_index = {}
        self.__default_index_name = '__index'
        self.vmvs = vmv_extensions
    
    @property
    def default_vmvs(self):
        """Returns the default list of Valued Missing Values."""
        if self._default_vmvs is None:
            self._default_vmvs = set([vmv.lower() if isinstance(vmv, str) else vmv for vmv in ["?", "UNK", "Unknown", "N/A", "NA", "", "(blank)"]])
            print(self._default_vmvs)
        return self._default_vmvs

    @property
    def vmvs(self):
        """Returns the extended Value Missing Values (default plus user provided)."""
        if not  self._vmvs:
            self._vmvs = self.default_vmvs
        return self._vmvs

    @vmvs.setter
    def vmvs(self, vmv_extensions: Optional[list] = []):
        """Allows extending default Valued Missing Values list, append only."""
        assert isinstance(vmv_extensions, list), "vmv extensions must be passed as a list"
        self._vmvs = self.default_vmvs.union(set([vmv.lower() if isinstance(vmv, str) else vmv for vmv in vmv_extensions]))

    @staticmethod
    def __get_flatline_index(column: pd.Series):
        """Returns an index for flatline events on a passed column.
        A flatline event is any full sequence of repeated values on the column.
        The returned index is a compact representation of all occurrences of flatline events.
        Returns a DataFrame with index equal to the index of first element in the event,
        a tail column identifying the last element of the sequence and a length column."""
        column.fillna('__filled')  # NaN values were not being treated as a sequence
        cum_differs_previous = column.ne(column.shift()).cumsum()
        sequence_groups = column.index.to_series().groupby(cum_differs_previous)
        data = {'length': sequence_groups.count().values,
        'ends': sequence_groups.last().values}
        flts = pd.DataFrame(data, index=sequence_groups.first().values).query('length > 1')
        return flts.rename_axis('starts')  # Adding index name for clarity

    def flatlines(self, th: int=5, skip: list=[]):
        """Checks the flatline index for flat sequences of length over a given threshold.
        Raises warning indicating columns with flatline events and total flatline events in the dataframe.
        Arguments:
            th: Defines the minimum length required for a flatline event to be reported.
            skip: List of columns that will not be target of search for flatlines.
                Pass '__index' in skip to skip looking for flatlines at the index."""
        df = self.df.copy()  # Index will not be covered in column iteration
        df[self.__default_index_name] = df.index  # Index now in columns to be processed next
        flatlines = {}
        for column in df.columns:  # Compile flatline index
            if column in skip:
                continue  # Column not requested
            flt_index = self._flatline_index.setdefault(column,
                self.__get_flatline_index(df[column]))
            flts = flt_index.loc[flt_index['length']>=th]
            if len(flts) > 0:
                flatlines[column] = flts
        if len(flatlines)>0:  # Flatlines detected
            total_flatlines = sum([flts.shape[0] for flts in flatlines.values()])
            self._warnings.add(
                QualityWarning(
                    test='Flatlines', category='Valued Missing Values', priority=2, data=flatlines,
                    description=f"Found {total_flatlines} flatline events with a minimun length of {th} among the columns {set(flatlines.keys())}."
            ))
            return flatlines
        else:
            print("[FLATLINES] No flatline events with a minimum length of {} were found.")

    def predefined_valued_missing_values(self, skip: list=[]):
        """Runs a check against a list of predefined Valued Missing Values.
        Raises warning based on the existence of these values.
        Returns a DataFrame with count distribution for each predefined type over each column.
        Arguments:
            skip: List of columns that will not be target of search for vmvs.
                Pass '__index' in skip to skip looking for flatlines at the index."""
        df = self.df.copy()  # Index will not be covered in column iteration
        df[self.__default_index_name] = df.index  # Index now in columns to be processed
        check_cols = set(df.columns).difference(set(skip))
        df = df[check_cols]
        vmvs = pd.DataFrame(index=self._vmvs, columns=check_cols)
        for vmv in self._vmvs:
            check_vmv = lambda x: x.lower()==vmv if isinstance(x,str) else x==vmv
            vmvs.loc[vmv] = df.applymap(check_vmv).sum()
        no_vmv_cols = vmvs.columns[vmvs.sum()==0]
        no_vmv_rows = vmvs.index[vmvs.sum(axis=1)==0]
        vmvs.drop(no_vmv_cols, axis=1, inplace=True)
        vmvs.drop(no_vmv_rows, inplace=True)
        if vmvs.empty:
            print("[PREDEFINED VALUED MISSING VALUES] No predefined vmvs from  the set {} were found in the dataset.".format(
                self.predefined_valued_missing_values
            ))
        else:
            total_vmvs = vmvs.sum().sum()
            self._warnings.add(
                QualityWarning(
                    test='Predefined Valued Missing Values', category='Valued Missing Values', priority=2, data=vmvs,
                    description=f"Found {total_vmvs} vmvs in the dataset."
            ))
            return vmvs