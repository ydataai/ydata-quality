"""
Implementation of Erroneous Data Identifier engine class to run erroneous data analysis.
"""

from typing import Optional

from pandas import DataFrame

from ..core import QualityEngine, QualityWarning
from ..utils.enum import DataFrameType


class ErroneousDataIdentifier(QualityEngine):
    "Engine for running analysis on erroneous data."

    def __init__(self, df: DataFrame, ed_extensions: Optional[list] = [], severity: Optional[str] = None):
        """
        Args:
            df (DataFrame): DataFrame used to run the erroneous data analysis.
            ed_extensions: A list of user provided erroneous data values to append to defaults.
            severity (str, optional): Sets the logger warning threshold to one of the valid levels [DEBUG, INFO, WARNING, ERROR, CRITICAL]
        """
        super().__init__(df=df, severity=severity)
        if self.df_type == DataFrameType.TIMESERIES:
            self._tests = ["flatlines", "predefined_erroneous_data"]
        else:
            self._tests = ["predefined_erroneous_data"]
        self._default_ed = None
        self._flatline_index = {}
        self.__default_index_name = '__index'
        self.err_data = ed_extensions

    @property
    def default_err_data(self):
        """Returns the default list of erroneous data values.
        ED values of string type are case insensitive during search."""
        if self._default_ed is None:
            self._default_ed = set([edv.lower() if isinstance(edv, str) else edv for edv in [
                                   "?", "UNK", "Unknown", "N/A", "NA", "", "(blank)"]])
        return self._default_ed

    @property
    def err_data(self):
        """Returns the extended erroneous data values (default plus user provided).
        ED values of string type are case insensitive during search."""
        if not self._edv:
            self._edv = self.default_err_data
        return self._edv

    @err_data.setter
    def err_data(self, err_data_extensions: Optional[list] = []):
        """Allows extending default erroneous data values list, append only.
        ED values of string type are case insensitive during search."""
        assert isinstance(err_data_extensions, list), "Erroneous data value extensions must be passed as a list."
        self._edv = self.default_err_data.union(
            set([edv.lower() if isinstance(edv, str) else edv for edv in err_data_extensions]))

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
            column = df[column_name]
            column.fillna('__filled')  # So NaN values are considered
            # Everytime shifted value is different from previous a new sequence starts
            sequence_indexes = column.ne(column.shift()).cumsum()
            sequence_groups = column.index.to_series().groupby(sequence_indexes)  # Group series indexes by sequence indexes
            data = {'length': sequence_groups.count().values,
                    'ends': sequence_groups.last().values}
            # Just dropping single unique values (detected as independent sequences)
            flts = DataFrame(data, index=sequence_groups.first().values).query('length > 1')
            flts.rename_axis('starts', inplace=True)  # Adding index name for clarity
            self._flatline_index[column_name] = flts  # Cache the index
        return flts.loc[flts['length'] >= th]

    def flatlines(self, th: int = 5, skip: list = []):
        """Iterates the dataset over columns and requests flatline indexes based on arguments.
        Raises warning indicating columns with flatline events and total flatline events in the dataframe.
        Arguments:
            th: Defines the minimum length required for a flatline event to be reported.
            skip: List of columns that will not be target of search for flatlines.
                Pass '__index' inside skip list to skip looking for flatlines at the index."""
        if self.df_type == DataFrameType.TABULAR:
            self._logger.debug('The provided DataFrame is not a valid Timeseries type, skipping flatlines test.')
            return None
        flatlines = {}
        for column in self.df.columns:  # Compile flatline index
            if column in skip or self.dtypes[column] == 'categorical':
                continue  # Column not requested or is categorical
            flts = self.__get_flatline_index(column, th)
            if len(flts) > 0:
                flatlines[column] = flts
        if len(flatlines) > 0:  # Flatlines detected
            total_flatlines = sum([flts.shape[0] for flts in flatlines.values()])
            self.store_warning(
                QualityWarning(
                    test='Flatlines', category='Erroneous Data', priority=2, data=flatlines,
                    description=f"Found {total_flatlines} flatline events \
                        with a minimun length of {th:.0f} among the columns {set(flatlines.keys())}."
            ))
            return flatlines
        
        self._logger.info(f"No flatline events with a minimum length of {th:.0f} were found.")

        return None

    def predefined_erroneous_data(self, skip: list = [], short: bool = True):
        """Runs a check against a list of predefined erroneous data values.
        Will always use the extended list if user provided any extension to the defaults.
        Raises warning based on the existence of these values.
        Erroneous data values of string type are case insensitive during search.
        Returns a DataFrame with count distribution for each predefined type over each column.
        Arguments:
            skip: List of columns that will not be target of search for predefined ED.
                Pass '__index' in skip to skip looking for flatlines at the index.
            short: Instruct engine to return only for ED values and columns where ED were detected"""
        df = self.df.copy()  # Index will not be covered in column iteration
        df[self.__default_index_name] = df.index  # Index now in columns to be processed
        check_cols = set(df.columns).difference(set(skip))
        df = df[check_cols]
        eds = DataFrame(index=self._edv, columns=check_cols)

        check_ed = lambda ed: lambda x: x.lower() == ed if isinstance(x, str) else x == ed
        for ed in self._edv:
            eds.loc[ed] = df.applymap(check_ed(ed)).sum()

        if short:
            no_ed_cols = eds.columns[eds.sum() == 0]
            no_ed_rows = eds.index[eds.sum(axis=1) == 0]
            eds.drop(no_ed_cols, axis=1, inplace=True)
            eds.drop(no_ed_rows, inplace=True)
        if eds.empty:
            self._logger.info("No predefined ED values from  the set %s were found in the dataset.", self.err_data)

            return None

        total_eds = eds.sum().sum()
        self.store_warning(
            QualityWarning(
                test='Predefined Erroneous Data', category='Erroneous Data', priority=2, data=eds,
                description=f"Found {total_eds} ED values in the dataset."
            ))
        return eds
