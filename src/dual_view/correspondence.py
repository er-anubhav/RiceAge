"""
Dual-View Correspondence Engine for RiceAge Multi-Scanner System.
Implements strict bipartite one-to-one grain matching with geometric distance gating,
multi-attribute cost optimization, ambiguity detection, and cluster segregation.
"""

from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from src.dual_view.models import ViewObservation, CorrespondenceStatus
from src.dual_view.registration import DualViewRegistration


class DualViewCorrespondenceEngine:
    """
    Executes safe one-to-one matching between bottom and top scanner observations.
    Guarantees:
      - Bipartite matching: 1 bottom <-> at most 1 top.
      - Geometric gating: pairs with distance > max_distance_px are never matched.
      - Multi-attribute cost: spatial distance + area discrepancy + aspect ratio.
      - Ambiguity detection: near-equidistant candidates flagged as AMBIGUOUS.
      - Unresolved clusters: large multi-grain blobs segregated as SEGMENTATION_UNRESOLVED.
    """

    def __init__(
        self,
        registration: DualViewRegistration,
        max_matching_distance_px: float = 35.0,
        ambiguity_margin_px: float = 8.0,
        max_hungarian_cost: float = 0.85,
        cost_spatial_weight: float = 0.50,
        cost_area_weight: float = 0.30,
        cost_aspect_ratio_weight: float = 0.20,
        cluster_area_threshold_px: float = 9000.0,
        cluster_solidity_threshold: float = 0.80
    ):
        self.registration = registration
        self.max_dist_px = max_matching_distance_px
        self.ambiguity_margin = ambiguity_margin_px
        self.max_cost = max_hungarian_cost
        self.w_spatial = cost_spatial_weight
        self.w_area = cost_area_weight
        self.w_ar = cost_aspect_ratio_weight
        self.cluster_area_thresh = cluster_area_threshold_px
        self.cluster_solidity_thresh = cluster_solidity_threshold

    def match_observations(
        self,
        bottom_obs: List[ViewObservation],
        top_obs: List[ViewObservation]
    ) -> List[Dict[str, Any]]:
        """
        Matches bottom and top observations.
        Returns a list of match records, each specifying:
          - 'bottom_obs': ViewObservation or None
          - 'top_obs': ViewObservation or None
          - 'status': CorrespondenceStatus
          - 'confidence': float
          - 'distance_px': float or None
          - 'cost_components': dict
        """
        # If registration is unhealthy, fail closed
        if not self.registration.is_healthy:
            return self._fail_closed_registration(bottom_obs, top_obs)

        # 1. Identify and isolate unresolved clusters
        b_normal, b_clusters = self._filter_clusters(bottom_obs)
        t_normal, t_clusters = self._filter_clusters(top_obs)

        # 2. Project bottom normal coordinates to top platen space
        pts_b = np.array([[obs.centroid_x, obs.centroid_y] for obs in b_normal], dtype=np.float64) if b_normal else np.empty((0, 2))
        pts_t = np.array([[obs.centroid_x, obs.centroid_y] for obs in t_normal], dtype=np.float64) if t_normal else np.empty((0, 2))

        if len(pts_b) == 0 or len(pts_t) == 0:
            # Handle empty platen case
            return self._handle_unpaired(b_normal, t_normal, b_clusters, t_clusters)

        success, proj_b = self.registration.project_batch_bottom_to_top(pts_b)
        if not success:
            return self._fail_closed_registration(bottom_obs, top_obs)

        tree_t = cKDTree(pts_t)
        tree_b_proj = cKDTree(proj_b)

        # 3. Detect Ambiguous Candidates
        ambiguous_b_indices: Set[int] = set()
        ambiguous_t_indices: Set[int] = set()

        for i, pb in enumerate(proj_b):
            near_t = tree_t.query_ball_point(pb, r=self.max_dist_px)
            if len(near_t) > 1:
                dists = sorted([float(np.linalg.norm(pb - pts_t[j])) for j in near_t])
                if (dists[1] - dists[0]) < self.ambiguity_margin:
                    ambiguous_b_indices.add(i)
                    for j in near_t:
                        ambiguous_t_indices.add(j)

        for j, pt in enumerate(pts_t):
            near_b = tree_b_proj.query_ball_point(pt, r=self.max_dist_px)
            if len(near_b) > 1:
                dists = sorted([float(np.linalg.norm(pt - proj_b[i])) for i in near_b])
                if (dists[1] - dists[0]) < self.ambiguity_margin:
                    ambiguous_t_indices.add(j)
                    for i in near_b:
                        ambiguous_b_indices.add(i)

        # 4. Construct Bipartite Cost Matrix
        N_b, N_t = len(b_normal), len(t_normal)
        INF_COST = 1000.0
        cost_matrix = np.full((N_b, N_t), INF_COST, dtype=np.float64)
        cost_details: Dict[Tuple[int, int], Dict[str, float]] = {}

        for i in range(N_b):
            if i in ambiguous_b_indices:
                continue  # Exclude ambiguous candidates from automatic bipartite pairing
            
            pb = proj_b[i]
            gb = b_normal[i]
            near_t = tree_t.query_ball_point(pb, r=self.max_dist_px)

            for j in near_t:
                if j in ambiguous_t_indices:
                    continue  # Exclude ambiguous candidates
                
                gt = t_normal[j]
                d_spatial = float(np.linalg.norm(pb - pts_t[j]))
                
                # Area difference cost
                max_area = max(gb.area_px, gt.area_px, 1.0)
                a_diff = abs(gb.area_px - gt.area_px) / max_area
                
                # Aspect ratio cost
                max_ar = max(gb.aspect_ratio, gt.aspect_ratio, 0.01)
                ar_diff = abs(gb.aspect_ratio - gt.aspect_ratio) / max_ar
                
                c_sp = self.w_spatial * (d_spatial / self.max_dist_px)
                c_ar = self.w_area * a_diff
                c_asp = self.w_ar * ar_diff
                total_cost = c_sp + c_ar + c_asp

                cost_matrix[i, j] = total_cost
                cost_details[(i, j)] = {
                    "distance_px": round(d_spatial, 2),
                    "cost_spatial": round(c_sp, 4),
                    "cost_area": round(c_ar, 4),
                    "cost_aspect_ratio": round(c_asp, 4),
                    "cost_total": round(total_cost, 4)
                }

        # 5. Solve Optimal One-to-One Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_b: Set[int] = set()
        matched_t: Set[int] = set()
        match_records: List[Dict[str, Any]] = []

        for r, c in zip(row_ind, col_ind):
            cost_val = cost_matrix[r, c]
            if cost_val < self.max_cost:
                # Valid one-to-one match
                matched_b.add(r)
                matched_t.add(c)
                details = cost_details.get((r, c), {})
                # Confidence is inverse of normalized cost (1.0 = perfect match, 0.0 = at threshold)
                conf = max(0.0, 1.0 - (cost_val / self.max_cost))

                match_records.append({
                    "bottom_obs": b_normal[r],
                    "top_obs": t_normal[c],
                    "status": CorrespondenceStatus.MATCHED,
                    "confidence": conf,
                    "distance_px": details.get("distance_px"),
                    "cost_components": details
                })

        # 6. Record Ambiguous Observations
        for i in sorted(ambiguous_b_indices):
            if i not in matched_b:
                match_records.append({
                    "bottom_obs": b_normal[i],
                    "top_obs": None,
                    "status": CorrespondenceStatus.AMBIGUOUS,
                    "confidence": 0.0,
                    "distance_px": None,
                    "cost_components": {"reason": "Multiple near candidates within ambiguity margin"}
                })
                matched_b.add(i)

        for j in sorted(ambiguous_t_indices):
            if j not in matched_t:
                match_records.append({
                    "bottom_obs": None,
                    "top_obs": t_normal[j],
                    "status": CorrespondenceStatus.AMBIGUOUS,
                    "confidence": 0.0,
                    "distance_px": None,
                    "cost_components": {"reason": "Multiple near candidates within ambiguity margin"}
                })
                matched_t.add(j)

        # 7. Record Unmatched Bottom Observations
        for i in range(N_b):
            if i not in matched_b:
                match_records.append({
                    "bottom_obs": b_normal[i],
                    "top_obs": None,
                    "status": CorrespondenceStatus.UNMATCHED_BOTTOM,
                    "confidence": 1.0,
                    "distance_px": None,
                    "cost_components": {"reason": "No admissible candidate within search radius or cost cutoff"}
                })

        # 8. Record Unmatched Top Observations
        for j in range(N_t):
            if j not in matched_t:
                match_records.append({
                    "bottom_obs": None,
                    "top_obs": t_normal[j],
                    "status": CorrespondenceStatus.UNMATCHED_TOP,
                    "confidence": 1.0,
                    "distance_px": None,
                    "cost_components": {"reason": "No admissible candidate within search radius or cost cutoff"}
                })

        # 9. Record Unresolved Clusters
        for bc in b_clusters:
            match_records.append({
                "bottom_obs": bc,
                "top_obs": None,
                "status": CorrespondenceStatus.SEGMENTATION_UNRESOLVED,
                "confidence": 0.0,
                "distance_px": None,
                "cost_components": {"reason": "Cluster area exceeds segmentation threshold"}
            })

        for tc in t_clusters:
            match_records.append({
                "bottom_obs": None,
                "top_obs": tc,
                "status": CorrespondenceStatus.SEGMENTATION_UNRESOLVED,
                "confidence": 0.0,
                "distance_px": None,
                "cost_components": {"reason": "Cluster area exceeds segmentation threshold"}
            })

        return match_records

    def _filter_clusters(self, obs_list: List[ViewObservation]) -> Tuple[List[ViewObservation], List[ViewObservation]]:
        """Separates normal grains from unresolved multi-grain clusters."""
        normal, clusters = [], []
        for obs in obs_list:
            solidity = obs.raw_features.get("solidity", 1.0)
            sep_status = obs.raw_features.get("separation_status", "")
            is_unresolved = (
                sep_status == "under_segmented_cluster" or
                obs.area_px > self.cluster_area_thresh or
                solidity < self.cluster_solidity_thresh
            )
            if is_unresolved:
                clusters.append(obs)
            else:
                normal.append(obs)
        return normal, clusters

    def _fail_closed_registration(
        self,
        bottom_obs: List[ViewObservation],
        top_obs: List[ViewObservation]
    ) -> List[Dict[str, Any]]:
        """Handles fail-closed behavior when calibration health fails."""
        records = []
        health = self.registration.get_health_report()
        reason = health.get("reason", "Registration health failure")

        for b in bottom_obs:
            records.append({
                "bottom_obs": b,
                "top_obs": None,
                "status": CorrespondenceStatus.REGISTRATION_FAILED,
                "confidence": 0.0,
                "distance_px": None,
                "cost_components": {"reason": reason}
            })
        for t in top_obs:
            records.append({
                "bottom_obs": None,
                "top_obs": t,
                "status": CorrespondenceStatus.REGISTRATION_FAILED,
                "confidence": 0.0,
                "distance_px": None,
                "cost_components": {"reason": reason}
            })
        return records

    def _handle_unpaired(
        self,
        b_normal: List[ViewObservation],
        t_normal: List[ViewObservation],
        b_clusters: List[ViewObservation],
        t_clusters: List[ViewObservation]
    ) -> List[Dict[str, Any]]:
        """Handles cases where one platen is completely empty."""
        records = []
        for b in b_normal:
            records.append({
                "bottom_obs": b,
                "top_obs": None,
                "status": CorrespondenceStatus.UNMATCHED_BOTTOM,
                "confidence": 1.0,
                "distance_px": None,
                "cost_components": {"reason": "Opposing platen has no grains"}
            })
        for t in t_normal:
            records.append({
                "bottom_obs": None,
                "top_obs": t,
                "status": CorrespondenceStatus.UNMATCHED_TOP,
                "confidence": 1.0,
                "distance_px": None,
                "cost_components": {"reason": "Opposing platen has no grains"}
            })
        for bc in b_clusters:
            records.append({
                "bottom_obs": bc,
                "top_obs": None,
                "status": CorrespondenceStatus.SEGMENTATION_UNRESOLVED,
                "confidence": 0.0,
                "distance_px": None,
                "cost_components": {"reason": "Cluster area exceeds segmentation threshold"}
            })
        for tc in t_clusters:
            records.append({
                "bottom_obs": None,
                "top_obs": tc,
                "status": CorrespondenceStatus.SEGMENTATION_UNRESOLVED,
                "confidence": 0.0,
                "distance_px": None,
                "cost_components": {"reason": "Cluster area exceeds segmentation threshold"}
            })
        return records
