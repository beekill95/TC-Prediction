from ensemble_boxes import weighted_boxes_fusion_3d
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import statistics
from tqdm.auto import tqdm


class DBScanClustering:
    def __init__(self, genesis_threshold: float = 0.5, dbscan_kwargs: dict = dict(eps=6, min_samples=2)) -> None:
        self._genesis_threshold = genesis_threshold
        self._dbscan_kwargs = dbscan_kwargs

    def create_clustering_data(self, genesis_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function will transform the given dataframe into another dataframe,
        with additional columns such that it is nicer to work with clustering algorithm.
        The output of this function will be fed into `perform_clustering()` function.

        Parameters
        ----------
        genesis_df: pd.DataFrame
            Pandas dataframe that should have these columns: `path`, `date`, `lat`, `lon`, and `pred`.
            It also assumes that this genesis data is from a year.
        """
        df = genesis_df.copy()
        df['genesis'] = df['pred'] >= self._genesis_threshold

        df = df.sort_values(['lat', 'lon', 'date'])
        days = list(df['date'].unique())
        df['days_since_May_1st'] = df['date'].apply(lambda d: days.index(d))
        df['days_scaled'] = df['days_since_May_1st'] * 1.5

        # Clustering is performed on genesis events only.
        return df[df['genesis']]

    def perform_clustering(self, genesis_clustering_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform clustering on the dataframe outputted from `create_clustering_data()`.
        """
        df = genesis_clustering_df.copy()
        dbscan = cluster.DBSCAN(**self._dbscan_kwargs)
        results = dbscan.fit_predict(df[['lat', 'lon', 'days_scaled']])
        df['cluster'] = results
        return df

    def count_genesis(self, genesis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Count the number of genesis after clustering.
        Notice that, unlike the above two functions where they work with dataframe of a year,
        this function works on all the given years.

        Parameters
        ----------
        genesis_df: pd.DataFrame
            Pandas dataframe that should have these columns: `path`, `date`, `lat`, `lon`, and `pred`.
        """
        years = genesis_df['date'].apply(lambda d: d.year)
        unique_years = np.unique(years.values)

        count = []
        for year in tqdm(unique_years, total=len(unique_years)):
            genesis_in_year_df = genesis_df[years == year]
            genesis_in_year_df = self.create_clustering_data(genesis_in_year_df)
            clusters = self.perform_clustering(genesis_in_year_df)['cluster']

            count.append({
                'year': year,
                'genesis': len(np.unique(clusters[clusters >= 0]))
            })

        return pd.DataFrame(count)


class WeightedFusedBoxesClustering:
    def __init__(self, genesis_threshold: float = 0.5, iou_threshold: float = 0.5, skip_box_threshold: float = 0.0) -> None:
        self._genesis_threshold = genesis_threshold
        self._iou_threshold = iou_threshold
        self._skip_box_threshold = skip_box_threshold

    def create_clustering_data(self, genesis_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function will transform the given dataframe into another dataframe,
        with additional columns such that it is nicer to work with clustering algorithm.
        The output of this function will be fed into `perform_clustering()` function.

        Parameters
        ----------
        genesis_df: pd.DataFrame
            Pandas dataframe that should have these columns: `path`, `date`, `lat`, `lon`, and `pred`.
            It also assumes that this genesis data is from a year.
        """
        df = genesis_df.copy()
        df['genesis'] = df['pred'] >= self._genesis_threshold

        df = df.sort_values(['lat', 'lon', 'date'])
        days = list(df['date'].unique())
        df['days_since_May_1st'] = df['date'].apply(lambda d: days.index(d))

        return df

    def perform_clustering(self, genesis_clustering_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform clustering on the dataframe outputted from `create_clustering_data()`.
        """
        C = WeightedFusedBoxesClustering

        df = genesis_clustering_df.copy()
        df = C.construct_3d_spatial_temporal_genesis_box(df)
        df = C.generate_box_coordinates(df)

        # Get list of boxes.
        boxes_list = []
        scores_list = []
        labels_list = []
        for _, row in df.iterrows():
            boxes_list.append([
                row['norm_lat_lower'],
                row['norm_lon_left'],
                row['norm_from'],
                row['norm_lat_upper'],
                row['norm_lon_right'],
                row['norm_to'],
            ])
            scores_list.append(statistics.mean(row['pred']))
            labels_list.append(1.)

        # Perform weighted fused boxes grouping.
        fused_boxes, probs, _ = weighted_boxes_fusion_3d(
            [boxes_list],
            [scores_list],
            [labels_list],
            weights=None,
            iou_thr=self._iou_threshold if self._iou_threshold is not None else 0.01,
            skip_box_thr=self._skip_box_threshold)

        # Only get boxes.
        fused_boxes = fused_boxes[probs >= 0.6]

        # Once we get the boxes,
        # find the index of fused_boxes that the original boxes belong to. 
        clustering_results = []
        for _, row in df.iterrows():
            _, _, fused_box_idx = C.intersected_with(row, fused_boxes)
            clustering_results.append(fused_box_idx)

        df['cluster'] = clustering_results
        return df

    def count_genesis(self, genesis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Count the number of genesis after clustering.
        Notice that, unlike the above two functions where they work with dataframe of a year,
        this function works on all the given years.

        Parameters
        ----------
        genesis_df: pd.DataFrame
            Pandas dataframe that should have these columns: `path`, `date`, `lat`, `lon`, and `pred`.
        """
        years = genesis_df['date'].apply(lambda d: d.year)
        unique_years = np.unique(years.values)

        count = []
        for year in tqdm(unique_years, total=len(unique_years)):
            genesis_in_year_df = genesis_df[years == year]
            genesis_in_year_df = self.create_clustering_data(genesis_in_year_df)
            clusters = self.perform_clustering(genesis_in_year_df)['cluster']
            # print(clusters.values)

            count.append({
                'year': year,
                'genesis': len(np.unique(clusters[~np.isnan(clusters.values)]))
            })

        return pd.DataFrame(count)

    @staticmethod
    def construct_3d_spatial_temporal_genesis_box(genesis_df_with_date: pd.DataFrame):
        def create_current_genesis_box(loc, row: pd.Series):
            return {
                'lat': loc[0],
                'lon': loc[1],
                'from': row['days_since_May_1st'],
                'to': row['days_since_May_1st'] + 1,
                'pred': [row['pred']],
                'date': [row['date']],
            }


        def update_current_genesis_box(cur_box, row: pd.Series):
            assert cur_box['to'] == row['days_since_May_1st']
            cur_box = {**cur_box}
            cur_box['to'] += 1
            cur_box['pred'].append(row['pred'])
            cur_box['date'].append(row['date'])
            return cur_box

        genesis_df = genesis_df_with_date.groupby(['lat', 'lon'])

        boxes = []
        for loc, rows in genesis_df:
            # The rows are ordered in ascending `days_since_May_1st`.
            cur_box = None
            rows = rows[rows['genesis']]
            for _, row in rows.iterrows():
                if cur_box is None:
                    # First row.
                    cur_box = create_current_genesis_box(loc, row)
                else:
                    try:
                        cur_box = update_current_genesis_box(cur_box, row)
                    except AssertionError:
                        # The date of the current box the not match with the current date.
                        boxes.append(cur_box)

                        # Create new box.
                        cur_box = create_current_genesis_box(loc, row)

        return pd.DataFrame(boxes)

    @staticmethod
    def generate_box_coordinates(df: pd.DataFrame):
        def scale_min_max(v, min_v, max_v):
            return (v - min_v) / (max_v - min_v)

        lat = df['lat'].values
        lon = df['lon'].values
        lat_min, lat_max = lat.min(), lat.max() + 30
        lon_min, lon_max = lon.min(), lon.max() + 30
        days_min, days_max = df['from'].values.min(), df['to'].values.max()

        df = df.copy()
        df['norm_lat_lower'] = df['lat'].apply(
            lambda l: scale_min_max(l, lat_min, lat_max))
        df['norm_lon_left'] = df['lon'].apply(
            lambda l: scale_min_max(l, lon_min, lon_max))
        df['norm_lat_upper'] = df['lat'].apply(
            lambda l: scale_min_max(l + 30, lat_min, lat_max))
        df['norm_lon_right'] = df['lon'].apply(
            lambda l: scale_min_max(l + 30, lon_min, lon_max))
        df['norm_from'] = df['from'].apply(
            lambda d: scale_min_max(d, days_min, days_max))
        df['norm_to'] = df['to'].apply(
            lambda d: scale_min_max(d, days_min, days_max))
        return df

    @staticmethod
    def calc_iou_3d(box: np.ndarray, boxes: np.ndarray):
        def calc_area(boxes: np.ndarray) -> np.ndarray:
            w = boxes[:, 3] - boxes[:, 0]
            h = boxes[:, 4] - boxes[:, 1]
            d = boxes[:, 5] - boxes[:, 2]
            return w * h * d

        # This function will calculate the IoU ratio between
        # `box` which is an array of shape [6]
        # and a list of `boxes` which has shape [N, 6].
        box = box[None, ...]

        # Find the intersected coordinates.
        x1_intersect = np.maximum(box[:, 0], boxes[:, 0])
        y1_intersect = np.maximum(box[:, 1], boxes[:, 1])
        z1_intersect = np.maximum(box[:, 2], boxes[:, 2])
        x2_intersect = np.minimum(box[:, 3], boxes[:, 3])
        y2_intersect = np.minimum(box[:, 4], boxes[:, 4])
        z2_intersect = np.minimum(box[:, 5], boxes[:, 5])

        # Find the intersected area.
        area_intersection = (
            np.maximum(x2_intersect - x1_intersect, 0.)
            * np.maximum(y2_intersect - y1_intersect, 0.)
            * np.maximum(z2_intersect - z1_intersect, 0.)).astype(np.float32)

        # Find the union area.
        area_union = calc_area(box) + calc_area(boxes) - area_intersection
        return area_intersection / area_union

    @staticmethod
    def intersected_with(box_row: pd.Series, merged_boxes: np.ndarray):
        C = WeightedFusedBoxesClustering

        # Construct a coordinates for box similar to that in the merged boxes:
        # x1, y1, z1, x2, y2, z2
        box = np.asarray([
            box_row['norm_lat_lower'],
            box_row['norm_lon_left'],
            box_row['norm_from'],
            box_row['norm_lat_upper'],
            box_row['norm_lon_right'],
            box_row['norm_to'],
        ])

        iou = C.calc_iou_3d(box, merged_boxes)
        positive_iou_mask = iou > 0.
        box_indices = np.where(positive_iou_mask)[0]
        positive_iou = iou[positive_iou_mask]
        return (box_indices,
                positive_iou,
                box_indices[np.argmax(positive_iou)] if len(positive_iou) else None)
