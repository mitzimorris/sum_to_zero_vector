import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Rook, W
from typing import (Dict, List)

def clean_nbs(
        nbs: Rook, target_indices: List[int], remove_indices: List[int]
        ) -> None:
    """
    Modify a neighbors graph by removing indices from the list of neighbors
    and resetting neighbors, weights lists accordingly.
    - param nbs: neighbor graph
    - param target_indices: list of node ids to process
    - param remove_indices: list of non-neighbor node ids
    """
    for node in target_indices:
        neighbors = nbs.neighbors[node]
        weights = nbs.weights[node]
        clean_neighbors_weights = [(neighbor, weight) for neighbor, weight in zip(neighbors, weights) 
                             if neighbor not in remove_indices]
        if clean_neighbors_weights:
            clean_neighbors, clean_weights = zip(*clean_neighbors_weights)
        else:
            clean_neighbors, clean_weights = [], []
        nbs.neighbors[node] = list(clean_neighbors)
        nbs.weights[node] = list(clean_weights)

def nyc_cleanup(nbs: Rook, gdf: gpd.GeoDataFrame) -> None:
    """
    Modify neighbor graph of NYC to remove neighbor pairs between
    Manhattan and other boroughs (Brooklyn, Queens).
    - nbs (libpysal weights): neighborhood graph based on gdf row indices
    - gdf (geopandas dataframe) census tract labels, geometry
    """
    manhattan_indices = gdf[gdf['BoroName'] == 'Manhattan'].index
    brooklyn_indices = gdf[gdf['BoroName'] == 'Brooklyn'].index
    queens_indices = gdf[gdf['BoroName'] == 'Queens'].index
    brooklyn_and_queens = brooklyn_indices.append(queens_indices)
    
    clean_nbs(nbs, manhattan_indices, brooklyn_and_queens)
    clean_nbs(nbs, brooklyn_indices, manhattan_indices)
    clean_nbs(nbs, queens_indices, manhattan_indices)

def nyc_sort_by_comp_size(nyc_gdf: gpd.GeoDataFrame) -> (Rook, gpd.GeoDataFrame, List[int]):
    """
    Correct neighborhood graph computed from CT2010 shapefiles so that Manhattan
    is its own components, then reorder gdf by component size, descending.
    Return sorted gdf and neighbor graph.
    - gdf (geopandas dataframe) census tract labels, geometry
    """
    nyc_nbs = Rook.from_dataframe(nyc_gdf, geom_col='geometry')

    # 1. edit graph - remove spurious connections, get component labels
    nyc_cleanup(nyc_nbs, nyc_gdf)
    nyc_nbs_tmp = W(nyc_nbs.neighbors, nyc_nbs.weights)
    unique, counts = np.unique(nyc_nbs_tmp.component_labels, return_counts=True)

    # 2. sort gdf 'comp_size' and reset the index to preserve alignment
    nyc_gdf['comp_id'] = nyc_nbs_tmp.component_labels
    sizes = nyc_gdf['comp_id'].value_counts()
    nyc_gdf['comp_size'] = nyc_gdf['comp_id'].map(sizes)
    nyc_gdf_sorted = nyc_gdf.sort_values(by='comp_size', ascending=False).reset_index(drop=True)
    nyc_gdf_sorted = nyc_gdf_sorted.set_geometry('geometry')

    # 3. sorting invalidates neighbor graph indexing, recompute neighbor graph
    nyc_nbs_tmp = Rook.from_dataframe(nyc_gdf_sorted, geom_col='geometry')
    nyc_cleanup(nyc_nbs_tmp, nyc_gdf_sorted)
    nyc_nbs_clean = W(nyc_nbs_tmp.neighbors, nyc_nbs_tmp.weights)

    # 4. relabel components
    nyc_gdf_sorted['comp_id'] = nyc_nbs_clean.component_labels
    return (nyc_nbs_clean, nyc_gdf_sorted, list(sizes.sort_values(ascending=False)))
