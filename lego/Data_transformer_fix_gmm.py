"""
DataTransformer module.
"""

from collections import namedtuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo',
    ['column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'],
)


class DataTransformer(object):
    """
    Data Transformer.

    Model continuous columns with a BayesianGMM and normalize them to a scalar between [-1, 1]
    and a vector. Discrete columns are encoded using a OneHotEncoder.
    """

    def __init__(
        self,
        max_clusters=10,
        weight_threshold=0.00001,
        grouped_continuous_columns=None,
        grouped_discrete_columns=None,
    ):
        """Create a data transformer.

        Args:
            max_clusters (int): Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float): Weight threshold for a Gaussian distribution to be kept.
            grouped_continuous_columns (Iterable[Iterable[str]]): Optional column groups that
                should share a single continuous transformer. Each group is trained on the
                combined values of the provided columns.
            grouped_discrete_columns (Iterable[Iterable[str]]): Optional column groups that share a
                single one-hot encoder. Each group is fitted on the combined categorical values of
                the provided columns.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold
        self._grouped_continuous_columns = (
            [list(group) for group in grouped_continuous_columns]
            if grouped_continuous_columns is not None
            else []
        )
        self._grouped_discrete_columns = (
            [list(group) for group in grouped_discrete_columns]
            if grouped_discrete_columns is not None
            else []
        )

    def _fit_continuous(self, data):
        """
        Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame): A dataframe containing a column.

        Returns:
            namedtuple: A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(
            missing_value_generation='from_column',
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold,
        )
        gm.fit(data, column_name)
        gm._trained_column_name = column_name
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type='continuous',
            transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components,
        )

    def _fit_continuous_group(self, raw_data, group_columns, group_index):
        """Train a shared Bayesian GMM for a group of continuous columns."""

        placeholder_name = f'__group__{group_index}'
        normalization_stats = {}
        normalized_frames = []
        for column in group_columns:
            series = raw_data[column].astype(float)
            mean = series.mean()
            std = series.std()
            if std == 0:
                std = 1.0
            normalization_stats[column] = (mean, std)
            normalized_series = (series - mean) / std
            normalized_frames.append(
                normalized_series.to_frame(name=placeholder_name)
            )

        stacked = pd.concat(normalized_frames, ignore_index=True)
        gm = ClusterBasedNormalizer(
            missing_value_generation='from_column',
            max_clusters=min(len(stacked), self._max_clusters),
            weight_threshold=self._weight_threshold,
        )
        gm.fit(stacked, placeholder_name)
        gm._trained_column_name = placeholder_name
        gm._shared_columns = tuple(group_columns)
        gm._group_normalization_stats = normalization_stats
        num_components = sum(gm.valid_component_indicator)

        return {
            'transform': gm,
            'output_info': [SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            'output_dimensions': 1 + num_components,
        }

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame): A dataframe containing a column.

        Returns:
            namedtuple: A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        ohe._trained_column_name = column_name
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type='discrete',
            transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories,
        )

    def _fit_discrete_group(self, raw_data, group_columns, group_index):
        """Train a shared one-hot encoder for a group of discrete columns."""

        placeholder_name = f'__discrete_group__{group_index}'
        stacked = pd.concat(
            [raw_data[[column]].rename(columns={column: placeholder_name}) for column in group_columns],
            ignore_index=True,
        )
        ohe = OneHotEncoder()
        ohe.fit(stacked, placeholder_name)
        ohe._trained_column_name = placeholder_name
        ohe._shared_columns = tuple(group_columns)
        num_categories = len(ohe.dummies)

        return {
            'transform': ohe,
            'output_info': [SpanInfo(num_categories, 'softmax')],
            'output_dimensions': num_categories,
        }

    def fit(
        self,
        raw_data,
        discrete_columns=(),
        grouped_continuous_columns=None,
        grouped_discrete_columns=None,
    ):
        """
        Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.

        This step also counts the #columns in matrix data and span information.

        Args:
            raw_data (pd.DataFrame or np.ndarray): Dataset to be transformed.
            discrete_columns (Iterable[str]): Columns to be treated as discrete.
            grouped_continuous_columns (Iterable[Iterable[str]]): Optional column groups that
                should share a single continuous transformer.
            grouped_discrete_columns (Iterable[Iterable[str]]): Optional column groups sharing a
                single one-hot encoder.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if grouped_continuous_columns is None:
            grouped_continuous_columns = [list(group) for group in self._grouped_continuous_columns]
        else:
            grouped_continuous_columns = [list(group) for group in grouped_continuous_columns]

        if grouped_discrete_columns is None:
            grouped_discrete_columns = [list(group) for group in self._grouped_discrete_columns]
        else:
            grouped_discrete_columns = [list(group) for group in grouped_discrete_columns]

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)
            grouped_continuous_columns = [
                [str(column) for column in group] for group in grouped_continuous_columns
            ]
            grouped_discrete_columns = [
                [str(column) for column in group] for group in grouped_discrete_columns
            ]
        else:
            grouped_continuous_columns = [list(group) for group in grouped_continuous_columns]
            grouped_discrete_columns = [list(group) for group in grouped_discrete_columns]

        self._grouped_continuous_columns = [list(group) for group in grouped_continuous_columns]
        self._grouped_discrete_columns = [list(group) for group in grouped_discrete_columns]

        discrete_columns = set(discrete_columns)

        missing_group_columns = {
            column
            for group in grouped_continuous_columns
            for column in group
            if column not in raw_data.columns
        }
        if missing_group_columns:
            missing = ', '.join(sorted(missing_group_columns))
            raise ValueError(f'Grouped continuous columns not found in data: {missing}')

        group_lookup = {}
        processed_groups = {}
        for group in grouped_continuous_columns:
            if not group:
                continue
            canonical = tuple(group)
            processed_groups[canonical] = list(group)
            for column in group:
                if column in group_lookup:
                    raise ValueError(
                        f"Column `{column}` appears in multiple grouped continuous definitions."
                    )
                group_lookup[column] = canonical

        overlapping_discrete = discrete_columns.intersection(group_lookup)
        if overlapping_discrete:
            overlap = ', '.join(sorted(overlapping_discrete))
            raise ValueError(
                f'Columns cannot be both discrete and grouped continuous: {overlap}'
            )

        missing_discrete_group_columns = {
            column
            for group in grouped_discrete_columns
            for column in group
            if column not in raw_data.columns
        }
        if missing_discrete_group_columns:
            missing = ', '.join(sorted(missing_discrete_group_columns))
            raise ValueError(f'Grouped discrete columns not found in data: {missing}')

        discrete_group_lookup = {}
        processed_discrete_groups = {}
        for group in grouped_discrete_columns:
            if not group:
                continue
            canonical = tuple(group)
            processed_discrete_groups[canonical] = list(group)
            for column in group:
                if column in discrete_group_lookup:
                    raise ValueError(
                        f"Column `{column}` appears in multiple grouped discrete definitions."
                    )
                if column in group_lookup:
                    raise ValueError(
                        f"Column `{column}` cannot be both grouped discrete and grouped continuous."
                    )
                discrete_group_lookup[column] = canonical
                discrete_columns.add(column)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        fitted_group_transforms = {}
        fitted_discrete_group_transforms = {}
        group_index = 0
        discrete_group_index = 0
        for column_name in raw_data.columns:
            if column_name in discrete_group_lookup:
                group_key = discrete_group_lookup[column_name]
                if group_key not in fitted_discrete_group_transforms:
                    fitted_discrete_group_transforms[group_key] = self._fit_discrete_group(
                        raw_data, processed_discrete_groups[group_key], discrete_group_index
                    )
                    discrete_group_index += 1

                group_transform_info = fitted_discrete_group_transforms[group_key]
                column_transform_info = ColumnTransformInfo(
                    column_name=column_name,
                    column_type='discrete',
                    transform=group_transform_info['transform'],
                    output_info=group_transform_info['output_info'],
                    output_dimensions=group_transform_info['output_dimensions'],
                )
            elif column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            elif column_name in group_lookup:
                group_key = group_lookup[column_name]
                if group_key not in fitted_group_transforms:
                    fitted_group_transforms[group_key] = self._fit_continuous_group(
                        raw_data, processed_groups[group_key], group_index
                    )
                    group_index += 1

                group_transform_info = fitted_group_transforms[group_key]
                column_transform_info = ColumnTransformInfo(
                    column_name=column_name,
                    column_type='continuous',
                    transform=group_transform_info['transform'],
                    output_info=group_transform_info['output_info'],
                    output_dimensions=group_transform_info['output_dimensions'],
                )
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_transform_info.transform
        trained_column_name = getattr(gm, '_trained_column_name', column_name)
        normalization_stats = getattr(gm, '_group_normalization_stats', {})
        if column_name in normalization_stats:
            mean, std = normalization_stats[column_name]
            # avoid modifying original data frame in place
            data = data.copy()
            data[column_name] = (data[column_name].astype(float) - mean) / std
        if trained_column_name != column_name:
            renamed_data = data.rename(columns={column_name: trained_column_name})
        else:
            renamed_data = data

        transformed = gm.transform(renamed_data)
        if trained_column_name != column_name:
            transformed = transformed.rename(
                columns={
                    col: col.replace(trained_column_name, column_name) for col in transformed.columns
                }
            )

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        column_name = data.columns[0]
        trained_column_name = getattr(ohe, '_trained_column_name', column_name)
        if trained_column_name != column_name:
            renamed_data = data.rename(columns={column_name: trained_column_name})
        else:
            renamed_data = data

        transformed = ohe.transform(renamed_data)
        if trained_column_name != column_name:
            transformed.columns = [
                col.replace(trained_column_name, column_name) for col in transformed.columns
            ]

        return transformed.to_numpy()

    def _synchronous_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns synchronous.

        Outputs a list with Numpy arrays.
        """
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        return column_data_list

    def _parallel_transform(self, raw_data, column_transform_info_list):
        """
        Take a Pandas DataFrame and transform columns in parallel.
        Outputs a list with Numpy arrays.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=-1)(processes)

    def transform(self, raw_data):
        """
        Take raw data and output a matrix data.
        """
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(
                raw_data, self._column_transform_info_list
            )
        else:
            column_data_list = self._parallel_transform(raw_data, self._column_transform_info_list)

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        trained_column_name = getattr(gm, '_trained_column_name', column_transform_info.column_name)
        output_columns = list(gm.get_output_sdtypes())
        if trained_column_name != column_transform_info.column_name:
            actual_columns = [
                column.replace(trained_column_name, column_transform_info.column_name)
                for column in output_columns
            ]
        else:
            actual_columns = output_columns

        # Create DataFrame with normalized value and component selection
        # The first column is the normalized value, the rest are one-hot encoded components
        data = pd.DataFrame(columns=actual_columns)
        data[actual_columns[0]] = column_data[:, 0]  # normalized value
        data[actual_columns[1]] = np.argmax(column_data[:, 1:], axis=1)  # component selection
        data = data.astype(float)
        
        if trained_column_name != column_transform_info.column_name:
            rename_map = {
                actual: original for actual, original in zip(actual_columns, output_columns)
            }
            data = data.rename(columns=rename_map)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        recovered = gm.reverse_transform(data)
        if trained_column_name != column_transform_info.column_name:
            recovered = recovered.rename(
                columns={trained_column_name: column_transform_info.column_name}
            )

        normalization_stats = getattr(gm, '_group_normalization_stats', {})
        if column_transform_info.column_name in normalization_stats:
            mean, std = normalization_stats[column_transform_info.column_name]
            recovered[column_transform_info.column_name] = (
                recovered[column_transform_info.column_name].astype(float) * std + mean
            )

        return recovered

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        trained_column_name = getattr(ohe, '_trained_column_name', column_transform_info.column_name)
        output_columns = list(ohe.get_output_sdtypes())
        if trained_column_name != column_transform_info.column_name:
            actual_columns = [
                column.replace(trained_column_name, column_transform_info.column_name)
                for column in output_columns
            ]
        else:
            actual_columns = output_columns

        data = pd.DataFrame(column_data, columns=actual_columns)
        if trained_column_name != column_transform_info.column_name:
            rename_map = {
                actual: original for actual, original in zip(actual_columns, output_columns)
            }
            data = data.rename(columns=rename_map)

        recovered = ohe.reverse_transform(data)
        if trained_column_name != column_transform_info.column_name:
            recovered = recovered.rename(
                columns={trained_column_name: column_transform_info.column_name}
            )

        return recovered[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        """
        Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
            self._column_raw_dtypes
        )
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """
        Get the ids of the given `column_name`.
        """

        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        trained_column_name = getattr(ohe, '_trained_column_name', column_transform_info.column_name)
        if trained_column_name != column_transform_info.column_name:
            data = data.rename(columns={column_transform_info.column_name: trained_column_name})

        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot),
        }

