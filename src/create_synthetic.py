import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from shapely.geometry import box, mapping, LineString
import json
import pickle
from src.create_mesh import create_mesh, fetch_outline, create_rectangle_mesh, get_bbox_from_outline
import icepack
import firedrake
from icepack.constants import (ice_density as ρ_I, water_density as ρ_W, gravity as g,)
import tqdm
from firedrake import inner, as_vector, assemble, grad, div, conditional, le, ge, And, Mesh, inner, Constant, sqrt
from icepack.calculus import FacetNormal
from icepack.models.hybrid import _pressure_approx
from operator import itemgetter
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import xgboost as xgb
import keras
#from firedrake import assemble, Constant, inner, grad, dx


class CreateSynthetic:

    def __init__(self):
        self.indicator = None

        # --- ML traction (tau_ml) state ---
        self.use_ml_drag = False
        self.ml_model = None
        self.ml_scaler_input = None
        self.ml_scaler_output = None
        self.ml_feature_names = None

        # ML traction as a 2D vector Constant (Pa)
        self.tau_ml_const = None

        # --- Constant ML feature bundle (set once per coarse run) ---
        self.ml_const_features = None
        self.real_surface_slope = None  # for reference

        self.training_bounds = None     # dict: {col: (lo, hi)}
        self.training_bounds_q = (0.01, 0.99)
        self.debug_tau = False        # enable/disable traction diagnostics
        self._tau_print_k = 0         # current outer iteration
        self._tau_printed_k = None    # last iteration already printed

    def load_training_bounds(
        self,
        train_csv="output_data/synthetic_ice_flow_parameters_test_coarse_real_surface_slope.csv",
        lo_q=0.01,
        hi_q=0.99,
        tau_x_col="Bed Vector Stress X (Pa)",
        tau_y_col="Bed Vector Stress Y (Pa)",
        tau_mag_key="Bed Vector Stress Mag (Pa)",
    ):
        """
        Cache per-feature quantile bounds from training data for simple OOD checks.
        Also caches traction magnitude bounds computed from (tau_x, tau_y).
        """

        df = pd.read_csv(train_csv)

        bounds = {}
        for col in self.ml_feature_names:
            if col not in df.columns:
                raise KeyError(f"Training CSV missing column: {col}")
            lo = float(df[col].quantile(lo_q))
            hi = float(df[col].quantile(hi_q))
            bounds[col] = (lo, hi)

        # --- NEW: magnitude bounds from vector components ---
        if tau_x_col in df.columns and tau_y_col in df.columns:
            tau_mag = np.sqrt(df[tau_x_col].to_numpy()**2 + df[tau_y_col].to_numpy()**2)
            bounds[tau_mag_key] = (
                float(np.quantile(tau_mag, lo_q)),
                float(np.quantile(tau_mag, hi_q)),
            )
            self.tau_mag_hi = bounds[tau_mag_key][1]
            # optional: keep some “cap candidates”
            self.training_tau_mag_stats = {
                "max": float(np.max(tau_mag)),
                "q99": float(np.quantile(tau_mag, 0.99)),
                "q999": float(np.quantile(tau_mag, 0.999)),
            }
        else:
            self.training_tau_mag_stats = None

        self.training_bounds = bounds
        self.training_bounds_q = (float(lo_q), float(hi_q))
        return bounds

    def compute_mean_velocities(self, u, h):
        if self.indicator is None:
            self.compute_box_indicator()
        ds = firedrake.ds_b

        A = firedrake.assemble(self.indicator * ds)
        A_val = float(A)
        if A_val == 0.0:
            raise ValueError("Indicator area on ds_b is zero.")

        # area-mean
        ux_A = firedrake.assemble(u[0] * self.indicator * ds) / A
        uy_A = firedrake.assemble(u[1] * self.indicator * ds) / A

        # thickness-weighted
        w = h * self.indicator
        denom = firedrake.assemble(w * ds)
        denom_val = float(denom)
        if denom_val == 0.0:
            raise ValueError("∫(h*indicator) dA is zero.")

        ux_h = firedrake.assemble(w * u[0] * ds) / denom
        uy_h = firedrake.assemble(w * u[1] * ds) / denom

        return (float(ux_A), float(uy_A)), (float(ux_h), float(uy_h))
            
    def compute_roughness_metrics(self, b):
        """
        RMS roughness (std of b) and slope roughness (RMS of |∇_xy b|)
        over the indicator region on the basal surface.
        """
        if self.indicator is None:
            self.compute_box_indicator()

        ds = firedrake.ds_b

        area = assemble(self.indicator * ds)
        area_val = float(area)
        if area_val <= 0.0:
            raise ValueError("Indicator area on ds_b is zero; check indicator/bbox.")

        mean_b = assemble(b * self.indicator * ds) / area

        rms_roughness = (assemble((b - mean_b)**2 * self.indicator * ds) / area)**0.5

        gb = firedrake.grad(b)
        gb_xy2 = gb[0]*gb[0] + gb[1]*gb[1]   # horizontal slope magnitude squared

        slope_roughness = (assemble(gb_xy2 * self.indicator * ds) / area)**0.5

        return float(rms_roughness), float(slope_roughness)
    
    def compute_mean_thickness(self, h):
        """
        Area-mean thickness over the indicator region (footprint).
        For your Q space (vfamily="R"), h is constant in z so ds_b == ds_t.
        """
        if self.indicator is None:
            self.compute_box_indicator()

        ds = firedrake.ds_b
        area = firedrake.assemble(self.indicator * ds)
        area_val = float(area)
        if area_val <= 0.0:
            raise ValueError("Indicator area on ds_b is zero; check bbox/indicator.")

        h_mean = firedrake.assemble(h * self.indicator * ds) / area
        return float(h_mean)
    
    def compute_mean_surface_height(self, s):
        """
        Area-mean surface height over the indicator region on the top surface.
        Returns: float (m)
        """
        if self.indicator is None:
            self.compute_box_indicator()

        ds = firedrake.ds_t
        area = firedrake.assemble(self.indicator * ds)
        area_val = float(area)
        if area_val <= 0.0:
            raise ValueError("Indicator area on ds_t is zero; check bbox/indicator.")

        s_mean = firedrake.assemble(s * self.indicator * ds) / area
        return float(s_mean)

    def compute_surface_slope_metrics(self, s):
        """
        Surface slope metrics over the indicator region on the top surface.

        Returns
        -------
        slope_vec_mean : (float, float)
            ( <ds/dx>, <ds/dy> ) area-mean over ds_t
        slope_rms : float
            sqrt( <|∇_xy s|^2> ) over ds_t
        """
        if self.indicator is None:
            self.compute_box_indicator()

        ds = firedrake.ds_t
        area = firedrake.assemble(self.indicator * ds)
        area_val = float(area)
        if area_val <= 0.0:
            raise ValueError("Indicator area on ds_t is zero; check bbox/indicator.")

        gs = firedrake.grad(s)
        ds_dx = gs[0]
        ds_dy = gs[1]
        slope2 = ds_dx*ds_dx + ds_dy*ds_dy

        mean_ds_dx = firedrake.assemble(ds_dx * self.indicator * ds) / area
        mean_ds_dy = firedrake.assemble(ds_dy * self.indicator * ds) / area
        slope_rms  = firedrake.sqrt(firedrake.assemble(slope2 * self.indicator * ds) / area)

        return (float(mean_ds_dx), float(mean_ds_dy)), float(slope_rms)

    # Used
    def set_ml_constant_features(
            self,
            *,
            latent_vector,
            rms_roughness,
            slope_roughness,
        ):
        """
        Store features that are held constant during the coarse-run ML coupling.

        surface_slope_equiv: the training-equivalent slope feature (scalar)
        latent_vector: array-like (latent_dim,) or (1, latent_dim)
        rms_roughness, slope_roughness: scalars
        """
        lv = np.asarray(latent_vector).reshape(-1)
        self.ml_const_features = {
            "RMS Roughness (m)": float(rms_roughness),
            "Slope Roughness": float(slope_roughness),
            "latent_vector": lv,  # stored separately; expanded into latent_i later
        }

    def predict_tau_ml_from_feature_dict(self, feature_dict):
        """
        Returns (tau_x, tau_y) in Pa from a one-row feature dict.
        """
        if not self.use_ml_drag:
            return 0.0, 0.0
        if self.ml_model is None or self.ml_scaler_input is None or self.ml_scaler_output is None:
            raise RuntimeError("ML model/scalers not loaded. Call load_ml_drag_model(...) first.")
        if self.ml_feature_names is None:
            raise RuntimeError("ml_feature_names not set. Provide input_features when loading model.")

        df = pd.DataFrame([feature_dict])
        X = df[self.ml_feature_names] #.to_numpy()
        Xs = self.ml_scaler_input.transform(X)

        y_pred = np.asarray(self.ml_model.predict(Xs))
        if y_pred.ndim == 1:
            # Some XGB setups return (2,) for a single row; coerce
            y_pred = y_pred.reshape(1, -1)

        if y_pred.shape[1] != 2:
            raise ValueError(f"Expected 2 outputs (tau_x,tau_y); got shape {y_pred.shape}")

        y_pa = self.ml_scaler_output.inverse_transform(y_pred)
        return float(y_pa[0, 0]), float(y_pa[0, 1])
    
    def _cellwise_boundary_mean_dg0(self, expr, *, which="b", weight=None, eps=1e-12, name_prefix="cellmean"):
        """
        Compute per-cell boundary mean of `expr` on ds_b or ds_t:
            mean_K = (∫_{Γ_*∩K} weight*expr dA) / (∫_{Γ_*∩K} weight dA)

        Parameters
        ----------
        expr : UFL scalar expression (e.g., h, s, grad(s)[0])
        which : {"b","t"} selects ds_b or ds_t
        weight : UFL scalar or None (if None, weight=1)
        Returns
        -------
        mean_arr : np.ndarray (ncells,)
        """

        if not hasattr(self, "Q_dg0"):
            raise RuntimeError("self.Q_dg0 not found. Initialize DG0 scalar space first.")

        Q0 = self.Q_dg0
        v = firedrake.TestFunction(Q0)
        ds = firedrake.ds_b if which == "b" else firedrake.ds_t

        if weight is None:
            weight = 1.0

        D = firedrake.Function(Q0, name=f"{name_prefix}_den_{which}")
        N = firedrake.Function(Q0, name=f"{name_prefix}_num_{which}")

        firedrake.assemble(weight * v * ds, tensor=D)
        firedrake.assemble(weight * expr * v * ds, tensor=N)

        D_arr = np.asarray(D.dat.data_ro)
        N_arr = np.asarray(N.dat.data_ro)

        # guard
        D_arr = D_arr + eps
        return (N_arr / D_arr).astype(float)

    def basal_mean_velocity_per_cell_dg0(self, u, h=None, *, thickness_weighted=False, eps=1e-12):
        """
        Compute per-cell mean horizontal velocity on the basal surface (ds_b), returned as DG0 arrays.

        This is the *cellwise* analogue of your fine "box average over ds_b" definition:
            ubar_K = (∫_{Γ_b∩K} u dA) / (∫_{Γ_b∩K} 1 dA)

        If thickness_weighted=True (and h is provided), computes:
            ubar_K = (∫_{Γ_b∩K} h*u dA) / (∫_{Γ_b∩K} h dA)

        Parameters
        ----------
        u : firedrake.Function
            Vector-valued horizontal velocity (u[0], u[1]).
        h : firedrake.Function or None
            Thickness (scalar). Required if thickness_weighted=True.
        thickness_weighted : bool
            If True, weight by thickness h on the basal surface.
        eps : float
            Small stabilizer for denominators.

        Returns
        -------
        ux_arr, uy_arr : (np.ndarray, np.ndarray)
            Arrays of shape (ncells,) in the ordering of self.Q_dg0 DOFs.
        """

        if not hasattr(self, "Q_dg0"):
            raise RuntimeError("self.Q_dg0 not found. Initialize DG0 scalar space (Q_dg0) first.")
        if len(getattr(u, "ufl_shape", ())) != 1:
            raise TypeError("u must be vector-valued (e.g., dim=2).")
        if thickness_weighted and h is None:
            raise ValueError("h must be provided when thickness_weighted=True.")

        Q0 = self.Q_dg0
        v = firedrake.TestFunction(Q0)
        ds = firedrake.ds_b

        # Denominator per cell on basal surface
        D = firedrake.Function(Q0, name="basal_denom_cell")
        if thickness_weighted:
            firedrake.assemble(h * v * ds, tensor=D)
        else:
            firedrake.assemble(v * ds, tensor=D)

        D_arr = np.asarray(D.dat.data_ro)
        if np.any(D_arr <= 0.0):
            # If some cells have zero basal area (shouldn't happen on a standard extruded mesh),
            # guard to avoid divide-by-zero.
            D_arr = D_arr + eps

        # Numerators per cell on basal surface
        Nx = firedrake.Function(Q0, name="basal_num_ux_cell")
        Ny = firedrake.Function(Q0, name="basal_num_uy_cell")

        if thickness_weighted:
            firedrake.assemble(h * u[0] * v * ds, tensor=Nx)
            firedrake.assemble(h * u[1] * v * ds, tensor=Ny)
        else:
            firedrake.assemble(u[0] * v * ds, tensor=Nx)
            firedrake.assemble(u[1] * v * ds, tensor=Ny)

        ux_arr = np.asarray(Nx.dat.data_ro) / D_arr
        uy_arr = np.asarray(Ny.dat.data_ro) / D_arr

        return ux_arr.astype(float), uy_arr.astype(float)

    def basal_mean_velocity_features_per_cell(self, u, h, *, eps=1e-12):
        """
        Returns:
        ux_area, uy_area : basal area-mean per cell
        ux_h,    uy_h    : basal thickness-weighted mean per cell
        """
        ux_area, uy_area = self.basal_mean_velocity_per_cell_dg0(u, thickness_weighted=False, eps=eps)
        ux_h, uy_h       = self.basal_mean_velocity_per_cell_dg0(u, h=h, thickness_weighted=True, eps=eps)
        return ux_area, uy_area, ux_h, uy_h

    def mean_thickness_per_cell_dg0(self, h, *, eps=1e-12):
        """
        Per-cell area-mean thickness over basal surface:
            hbar_K = (∫_{Γ_b∩K} h dA) / (∫_{Γ_b∩K} 1 dA)

        Returns: h_arr shape (ncells,)
        """

        if not hasattr(self, "Q_dg0"):
            raise RuntimeError("self.Q_dg0 not found. Initialize DG0 scalar space first.")

        Q0 = self.Q_dg0
        v = firedrake.TestFunction(Q0)
        ds = firedrake.ds_b

        # area per cell on basal surface
        A = firedrake.Function(Q0, name="A_basal_cell")
        firedrake.assemble(v * ds, tensor=A)
        A_arr = np.asarray(A.dat.data_ro)
        if np.any(A_arr <= 0.0):
            A_arr = A_arr + eps

        # numerator per cell: ∫ h dA
        N = firedrake.Function(Q0, name="Nh_basal_cell")
        firedrake.assemble(h * v * ds, tensor=N)
        N_arr = np.asarray(N.dat.data_ro)

        h_arr = N_arr / A_arr
        return h_arr.astype(float)

    def mean_surface_height_per_cell_dg0(self, s, *, eps=1e-12):
        """
        Per-cell area-mean surface height over top surface:
            sbar_K = (∫_{Γ_t∩K} s dA) / (∫_{Γ_t∩K} 1 dA)

        Returns: s_arr shape (ncells,)
        """

        if not hasattr(self, "Q_dg0"):
            raise RuntimeError("self.Q_dg0 not found. Initialize DG0 scalar space first.")

        Q0 = self.Q_dg0
        v = firedrake.TestFunction(Q0)
        ds = firedrake.ds_t

        # area per cell on top surface
        A = firedrake.Function(Q0, name="A_top_cell")
        firedrake.assemble(v * ds, tensor=A)
        A_arr = np.asarray(A.dat.data_ro)
        if np.any(A_arr <= 0.0):
            A_arr = A_arr + eps

        # numerator per cell: ∫ s dA
        N = firedrake.Function(Q0, name="Ns_top_cell")
        firedrake.assemble(s * v * ds, tensor=N)
        N_arr = np.asarray(N.dat.data_ro)

        s_arr = N_arr / A_arr
        return s_arr.astype(float)
    
    def surface_slope_metrics_per_cell_dg0(self, s, *, eps=1e-12):
        """
        Per-cell surface slope metrics over top surface (ds_t):

        mean slope components:
            <ds/dx>_K = (∫_{Γ_t∩K} (ds/dx) dA) / A_K
            <ds/dy>_K = (∫_{Γ_t∩K} (ds/dy) dA) / A_K

        RMS slope magnitude:
            slope_rms_K = sqrt( (∫_{Γ_t∩K} (sx^2+sy^2) dA) / A_K )

        Returns:
        dsdx_arr, dsdy_arr, slope_rms_arr : arrays shape (ncells,)
        """

        if not hasattr(self, "Q_dg0"):
            raise RuntimeError("self.Q_dg0 not found. Initialize DG0 scalar space first.")

        Q0 = self.Q_dg0
        v = firedrake.TestFunction(Q0)
        ds = firedrake.ds_t

        # area per cell on top surface
        A = firedrake.Function(Q0, name="A_top_cell")
        firedrake.assemble(v * ds, tensor=A)
        A_arr = np.asarray(A.dat.data_ro)
        if np.any(A_arr <= 0.0):
            A_arr = A_arr + eps

        gs = firedrake.grad(s)
        sx = gs[0]
        sy = gs[1]
        slope2 = sx*sx + sy*sy

        # mean slope components per cell
        Nx = firedrake.Function(Q0, name="Nsx_top_cell")
        Ny = firedrake.Function(Q0, name="Nsy_top_cell")
        firedrake.assemble(sx * v * ds, tensor=Nx)
        firedrake.assemble(sy * v * ds, tensor=Ny)

        dsdx_arr = np.asarray(Nx.dat.data_ro) / A_arr
        dsdy_arr = np.asarray(Ny.dat.data_ro) / A_arr

        # RMS slope magnitude per cell
        N2 = firedrake.Function(Q0, name="Nslope2_top_cell")
        firedrake.assemble(slope2 * v * ds, tensor=N2)
        slope_rms_arr = np.sqrt(np.asarray(N2.dat.data_ro) / A_arr)

        return dsdx_arr.astype(float), dsdy_arr.astype(float), slope_rms_arr.astype(float)
    
    def update_tau_ml_field_from_state_dg0(
        self,
        *,
        temperature_K,
        C_value,
        u,
        h,
        s,
    ):
        """
        Predict one tau per coarse cell (DG0), using the *exact* feature schema used in training:

        columns = [
        'Thickness (m)', 'Surface Height (m)', 'Surface Slope X Y', 'Surface Slope',
        'Inflow Velocity (m/a)', 'Temperature (K)', 'C',
        'RMS Roughness (m)', 'Slope Roughness',
        'x-Mean Velocity (m/a)', 'y-Mean Velocity (m/a)',
        'x-Mean Velocity h (m/a)', 'y-Mean Velocity h (m/a)',
        latent_*
        ]

        Notes
        -----
        - Thickness and velocities are computed as basal-surface *cellwise means* (ds_b).
        - Surface height and slope are computed as top-surface *cellwise means* (ds_t).
        - Roughness + latent vector + inflow velocity are treated as constants (as in training).
        - Writes into self.tau_ml (DG0 vector Function).
        """

        # --- checks ---
        if self.ml_const_features is None:
            raise RuntimeError("ml_const_features not set. Call set_ml_constant_features(...) first.")
        if self.ml_model is None or self.ml_scaler_input is None or self.ml_scaler_output is None:
            raise RuntimeError("ML model/scalers not loaded. Call load_ml_drag_model(...) first.")
        if self.ml_feature_names is None:
            raise RuntimeError("ml_feature_names not set.")
        if not hasattr(self, "Q_dg0") or not hasattr(self, "V_dg0"):
            raise RuntimeError("DG0 spaces not initialized. Add Q_dg0/V_dg0 in create_function_space().")

        if getattr(self, "tau_ml", None) is None:
            self.tau_ml = firedrake.Function(self.V_dg0, name="tau_ml")
            self.tau_ml.assign(firedrake.Constant((0.0, 0.0)))

        eps = 1e-12

        # --- per-cell features (match your fine definitions) ---

        # Thickness mean over basal surface (ds_b)
        h_arr = self._cellwise_boundary_mean_dg0(h, which="b", weight=None, eps=eps, name_prefix="hmean")

        # Surface height mean over top surface (ds_t)
        s_arr = self._cellwise_boundary_mean_dg0(s, which="t", weight=None, eps=eps, name_prefix="smean")

        # Surface slope metrics over top surface (ds_t): mean ds/dx, mean ds/dy, and RMS |∇s|
        gs = firedrake.grad(s)
        sx = gs[0]
        sy = gs[1]
        slope2 = sx*sx + sy*sy

        dsdx_arr = self._cellwise_boundary_mean_dg0(sx, which="t", weight=None, eps=eps, name_prefix="dsdx")
        dsdy_arr = self._cellwise_boundary_mean_dg0(sy, which="t", weight=None, eps=eps, name_prefix="dsdy")

        # RMS slope magnitude per cell on ds_t: sqrt(mean(sx^2+sy^2))
        slope2_mean = self._cellwise_boundary_mean_dg0(slope2, which="t", weight=None, eps=eps, name_prefix="slope2")
        slope_rms_arr = np.sqrt(np.maximum(slope2_mean, 0.0))

        # NOT USING THIS BECAUSE:
        # Your training column "Surface Slope X Y" is ambiguous as a single column.
        # You currently store "mean_s_slope_x_y" as one item in list_of_parameters.
        # Best interpretation: it's the *vector mean* packaged somehow.
        # For strict reproducibility, store it as a single scalar in coarse too.
        #
        # I recommend defining it as the magnitude of the mean slope vector:
        #     |<∇s>| = sqrt(<sx>^2 + <sy>^2)
        #
        # If your training used something else, change ONLY this line to match it.
        slope_xy_arr = np.sqrt(dsdx_arr*dsdx_arr + dsdy_arr*dsdy_arr)

        # Basal velocities (ds_b): area-mean and thickness-weighted mean
        ux_area, uy_area, ux_h, uy_h = self.basal_mean_velocity_features_per_cell(u, h, eps=eps)

        # --- constants replicated per cell ---
        n_cells = len(h_arr)

        #u_in = float(self.ml_const_features["Inflow Velocity (m/a)"])  # must be set in ml_const_features
        rms_rough = float(self.ml_const_features["RMS Roughness (m)"])
        slope_rough = float(self.ml_const_features["Slope Roughness"])

        lv = np.asarray(self.ml_const_features["latent_vector"]).reshape(-1)

        # --- build DataFrame in your training column names ---
        X = pd.DataFrame({
            "Thickness (m)": h_arr,
            "Surface Height (m)": s_arr,
            "Surface Slope X Y": slope_xy_arr,
            "Surface Slope": slope_rms_arr,
            #"Inflow Velocity (m/a)": np.full(n_cells, u_in, dtype=float),
            "Temperature (K)": np.full(n_cells, float(temperature_K), dtype=float),
            "C": np.full(n_cells, float(C_value), dtype=float),
            "RMS Roughness (m)": np.full(n_cells, rms_rough, dtype=float),
            "Slope Roughness": np.full(n_cells, slope_rough, dtype=float),
            "x-Mean Velocity (m/a)": ux_area.astype(float),
            "y-Mean Velocity (m/a)": uy_area.astype(float),
            "x-Mean Velocity h (m/a)": ux_h.astype(float),
            "y-Mean Velocity h (m/a)": uy_h.astype(float),
        })

        for j, val in enumerate(lv):
            X[f"latent_{j}"] = float(val)

        # Enforce training column order
        X = X[self.ml_feature_names]

        # --- OOD bounds (unchanged from your code) ---
        if getattr(self, "training_bounds", None) is None:
            self.load_training_bounds()

        inside_all = np.ones(len(X), dtype=bool)
        summary = []
        for col in self.ml_feature_names:
            lo, hi = self.training_bounds[col]
            v = X[col].to_numpy()
            below = v < lo
            above = v > hi
            oob = below | above
            inside_all &= ~oob

            n_below = int(below.sum())
            n_above = int(above.sum())
            n_oob = n_below + n_above
            vmin = float(np.min(v))
            vmax = float(np.max(v))
            worst_below = float(lo - vmin) if n_below > 0 else 0.0
            worst_above = float(vmax - hi) if n_above > 0 else 0.0
            summary.append((col, n_oob, n_below, n_above, worst_below, worst_above))

        frac_inside_all = float(inside_all.mean())

        k = getattr(self, "_tau_print_k", None)
        should_print = True
        if k is not None:
            if getattr(self, "_tau_printed_k_ood", None) == k:
                should_print = False
            else:
                self._tau_printed_k_ood = k

        if should_print:
            summary.sort(key=lambda t: t[1], reverse=True)
            top = summary[:8]
            lo_q, hi_q = getattr(self, "training_bounds_q", (None, None))
            hdr = "OOD(simple quantile bounds): " if lo_q is None else f"OOD(simple {lo_q:.2g}-{hi_q:.2g} quantiles): "
            print(f"{hdr}{100*frac_inside_all:.2f}% CELLS inside all feature bounds")
            for col, n_oob, n_below, n_above, worst_below, worst_above in top:
                if n_oob == 0:
                    continue
                print(
                    f"  {col}: oob={n_oob}/{len(X)} "
                    f"(below={n_below}, above={n_above}); "
                    f"worst_below={worst_below:.3g}, worst_above={worst_above:.3g}"
                )

        # --- predict and write DG0 traction ---
        Xs = self.ml_scaler_input.transform(X)
        y_scaled = np.asarray(self.ml_model.predict(Xs))
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 2)

        y_pa = self.ml_scaler_output.inverse_transform(y_scaled)

        self.tau_ml.dat.data[:] = y_pa.astype(float)

        tx = y_pa[:, 0]
        ty = y_pa[:, 1]
        return float(tx.mean()), float(ty.mean()), float(tx.std()), float(ty.std())

    # Used
    # def update_tau_ml_field_from_state_dg0(
    #     self,
    #     *,
    #     temperature_K,
    #     C_value,
    #     u,
    #     h,
    # ):
    #     """
    #     predict one tau per coarse cell (DG0).
    #     Uses per-cell thickness and thickness-weighted velocity:
    #         ubar_cell = avg(h*u)/avg(h)
    #     Writes into self.tau_ml (DG0 vector Function).

    #     Also prints a directional OOD summary showing how many CELLS fall outside
    #     the training 1–99% (or chosen) quantile bounds per feature, plus worst excursions.
    #     """
    #     import numpy as np
    #     import pandas as pd
    #     import firedrake

    #     if self.ml_const_features is None:
    #         raise RuntimeError("ml_const_features not set. Call set_ml_constant_features(...) first.")
    #     if self.ml_model is None or self.ml_scaler_input is None or self.ml_scaler_output is None:
    #         raise RuntimeError("ML model/scalers not loaded. Call load_ml_drag_model(...) first.")
    #     if self.ml_feature_names is None:
    #         raise RuntimeError("ml_feature_names not set.")
    #     if not hasattr(self, "Q_dg0") or not hasattr(self, "V_dg0"):
    #         raise RuntimeError("DG0 spaces not initialized. Add Q_dg0/V_dg0 in create_function_space().")
    #     if getattr(self, "tau_ml", None) is None:
    #         self.tau_ml = firedrake.Function(self.V_dg0, name="tau_ml")
    #         self.tau_ml.assign(firedrake.Constant((0.0, 0.0)))

    #     # Cellwise mean thickness (DG0): avg(h) per cell
    #     h0 = firedrake.project(h, self.Q_dg0)

    #     # Cellwise mean of (h*u) (DG0 vector): avg(h*u) per cell
    #     hu0 = firedrake.project(h * u, self.V_dg0)

    #     h_arr = np.asarray(h0.dat.data_ro).reshape(-1)   # (ncells,)
    #     hu_arr = np.asarray(hu0.dat.data_ro)             # (ncells,2)

    #     # Thickness-weighted cell mean velocity
    #     eps = 1e-12
    #     ux_arr = hu_arr[:, 0] / (h_arr + eps)
    #     uy_arr = hu_arr[:, 1] / (h_arr + eps)

    #     lv = np.asarray(self.ml_const_features["latent_vector"]).reshape(-1)

    #     X = pd.DataFrame({
    #         "Thickness (m)": h_arr.astype(float),
    #         "Surface Slope (m)": float(self.ml_const_features["Surface Slope (m)"]),
    #         "Temperature (K)": float(temperature_K),
    #         "C": float(C_value),
    #         "RMS Roughness (m)": float(self.ml_const_features["RMS Roughness (m)"]),
    #         "Slope Roughness": float(self.ml_const_features["Slope Roughness"]),
    #         "x-Mean Velocity (m/a)": ux_arr.astype(float),
    #         "y-Mean Velocity (m/a)": uy_arr.astype(float),
    #     })

    #     for j, val in enumerate(lv):
    #         X[f"latent_{j}"] = float(val)

    #     # Enforce training column order
    #     X = X[self.ml_feature_names]

    #     # --- Directional OOD summary (per-feature, per-cell) ---
    #     if getattr(self, "training_bounds", None) is None:
    #         # Expected to set:
    #         #   self.training_bounds[col] = (lo, hi) for each feature col
    #         #   self.training_bounds_q = (lo_q, hi_q)
    #         # and ideally also self.tau_mag_hi, but we don't require it here.
    #         self.load_training_bounds()

    #     n = len(X)
    #     inside_all = np.ones(n, dtype=bool)
    #     summary = []

    #     for col in self.ml_feature_names:
    #         lo, hi = self.training_bounds[col]
    #         v = X[col].to_numpy()

    #         below = v < lo
    #         above = v > hi
    #         oob = below | above

    #         inside_all &= ~oob

    #         n_below = int(below.sum())
    #         n_above = int(above.sum())
    #         n_oob = n_below + n_above

    #         vmin = float(np.min(v))
    #         vmax = float(np.max(v))
    #         worst_below = float(lo - vmin) if n_below > 0 else 0.0  # positive magnitude below
    #         worst_above = float(vmax - hi) if n_above > 0 else 0.0  # positive magnitude above

    #         summary.append((col, n_oob, n_below, n_above, worst_below, worst_above))

    #     frac_inside_all = float(inside_all.mean())

    #     # Print compact summary: top offenders by total OOB count.
    #     # Gate printing to once per outer update if caller sets self._tau_print_k;
    #     # otherwise it prints every call.
    #     k = getattr(self, "_tau_print_k", None)
    #     should_print = True
    #     if k is not None:
    #         if getattr(self, "_tau_printed_k_ood", None) == k:
    #             should_print = False
    #         else:
    #             self._tau_printed_k_ood = k

    #     if should_print:
    #         summary.sort(key=lambda t: t[1], reverse=True)
    #         topk = 8
    #         top = summary[:topk]

    #         lo_q, hi_q = getattr(self, "training_bounds_q", (None, None))
    #         if lo_q is None or hi_q is None:
    #             hdr = "OOD(simple quantile bounds): "
    #         else:
    #             hdr = f"OOD(simple {lo_q:.2g}-{hi_q:.2g} quantiles): "

    #         print(f"{hdr}{100*frac_inside_all:.2f}% CELLS inside all feature bounds")

    #         for col, n_oob, n_below, n_above, worst_below, worst_above in top:
    #             if n_oob == 0:
    #                 continue
    #             print(
    #                 f"  {col}: oob={n_oob}/{n} "
    #                 f"(below={n_below}, above={n_above}); "
    #                 f"worst_below={worst_below:.3g}, worst_above={worst_above:.3g}"
    #             )

    #     # --- Predict and write DG0 traction ---
    #     Xs = self.ml_scaler_input.transform(X.values)
    #     y_scaled = self.ml_model.predict(Xs)  # (ncells,2) ideally
    #     y_scaled = np.asarray(y_scaled)

    #     if y_scaled.ndim == 1:
    #         # if model returns flattened, reshape
    #         y_scaled = y_scaled.reshape(-1, 2)

    #     y_pa = self.ml_scaler_output.inverse_transform(y_scaled)

    #     # write DG0 traction
    #     self.tau_ml.dat.data[:] = y_pa.astype(float)

    #     tx = y_pa[:, 0]
    #     ty = y_pa[:, 1]
    #     return float(tx.mean()), float(ty.mean()), float(tx.std()), float(ty.std())

    # used
    def load_ml_drag_model(self, model_path="xgb_shear_model", input_features=None):
        """
        Load saved XGBoost model + scalers that predict (tau_x, tau_y) in Pa.

        input_features must match training order exactly.
        """
        if input_features is None:
            raise ValueError("input_features must be provided in the exact training order.")

        model = xgb.XGBRegressor()
        model.load_model(f"{model_path}.json")

        scaler_input = joblib.load(f"{model_path}_scaler_input.pkl")
        scaler_output = joblib.load(f"{model_path}_scaler_output.pkl")

        self.ml_model = model
        self.ml_scaler_input = scaler_input
        self.ml_scaler_output = scaler_output
        self.ml_feature_names = list(input_features)

        self.use_ml_drag = True
        return model, scaler_input, scaler_output
    
    def set_tau_ml_const(self, tau_x_pa, tau_y_pa):
        """Set the (tau_x, tau_y) ML traction Constant in Pa."""
        if self.tau_ml_const is None:
            self.tau_ml_const = firedrake.Constant((0.0, 0.0))
        self.tau_ml_const.assign(firedrake.as_vector((float(tau_x_pa), float(tau_y_pa))))

    #used
    def coarse_diagnostic_solve_with_ml_updates(
        self,
        *,
        u_init,
        h,
        s,
        A,
        C,
        b,
        temperature_K,
        C_value,
        n_updates=10,
        use_indicator_mean=True,   # kept for API compatibility (not used in field updater)
        tol_tau=1e-2,
        verbose=True,
        update_use_ml_drag=True,
        # --- NEW knobs for Option B ---
        ramp_schedule=None,        # e.g., [0.05, 0.10, 0.20, 0.40, 0.80, 1.00]
        ramp_power=2.0,            # used only if ramp_schedule is None
        alpha0=0.05,               # used only if ramp_schedule is None; keeps k=0 mostly physics-only
        alpha_max=1.0,             # used only if ramp_schedule is None
    ):
        """
        Outer fixed-point (Picard) coupling with spatially varying ML traction field.

        Option B implemented (recommended): under-relaxation / blending
            tau_used^k = (1-alpha_k) * tau_used^(k-1) + alpha_k * tau_pred^k

        For k=0 we start from ~physics-only by applying only a small fraction of the
        predicted traction:
            tau_used^0 = alpha_0 * tau_pred^0   with alpha_0 small (default 0.05)

        tol_tau stopping criterion is on the *applied* traction field (Pa):
            max_i ||tau_used^k(i) - tau_used^(k-1)(i)|| < tol_tau
        """
        import numpy as np

        if not getattr(self, "use_ml_drag", False):
            raise RuntimeError("use_ml_drag is False. Load ML model first (load_ml_drag_model).")
        if self.ml_const_features is None:
            raise RuntimeError("ml_const_features not set. Call set_ml_constant_features(...) first.")
        if getattr(self, "tau_ml", None) is None:
            raise RuntimeError("tau_ml is None. Initialize self.tau_ml as a Firedrake vector Function before calling.")

        def _alpha_k(k: int) -> float:
            """Return ramp/relaxation factor alpha_k in [0, 1]."""
            if ramp_schedule is not None:
                if len(ramp_schedule) == 0:
                    return 1.0
                return float(ramp_schedule[k]) if k < len(ramp_schedule) else float(ramp_schedule[-1])
            # Smooth ramp reaching 1 at last planned update, with small alpha at k=0.
            if n_updates <= 1:
                return 1.0
            t = (k + 1) / float(n_updates)  # in (0,1]
            a = alpha0 + (1.0 - alpha0) * (t ** float(ramp_power))
            # First clamp to [0,1] for safety
            a = float(min(1.0, max(0.0, a)))
            # Then cap to alpha_max (<=1)
            a = float(min(float(alpha_max), a))
            return a

        u = u_init

        # prev_tau will store the *applied* traction field DOFs (tau_used), shape (ndof, 2)
        prev_tau = None

        # return scalar summaries (means of predicted tau from ML updater, as before)
        last_tx_mean = None
        last_ty_mean = None

        for k in range(int(n_updates)):

            if update_use_ml_drag:
                self._tau_print_k = k

                # --- ML predicts tau field into self.tau_ml (call it tau_pred^k) ---
                # tx_mean, ty_mean, tx_std, ty_std = self.update_tau_ml_field_from_state_local(
                #     temperature_K=temperature_K,
                #     C_value=C_value,
                #     u=u,
                #     h=h,
                # )
                tx_mean, ty_mean, tx_std, ty_std = self.update_tau_ml_field_from_state_dg0(
                        temperature_K=temperature_K,
                        C_value=C_value,
                        u=u,
                        h=h,
                        s=s,
                    )
                last_tx_mean, last_ty_mean = tx_mean, ty_mean

                tau_pred = self.tau_ml.dat.data_ro.copy()  # raw ML prediction: tau_hat^(k)
                alpha = _alpha_k(k)

                # --- Option B: blend from previous applied tau (physics-only start) ---
                if prev_tau is None:
                    # Start from ~physics-only: apply only a small fraction of the ML prediction
                    tau_used = alpha * tau_pred
                    d_tau_max = float("inf")
                else:
                    tau_used = (1.0 - alpha) * prev_tau + alpha * tau_pred
                    d_tau = tau_used - prev_tau
                    d_tau_max = float(np.max(np.sqrt(d_tau[:, 0] ** 2 + d_tau[:, 1] ** 2)))

                # --- magnitude clip using training envelope ---
                cap = self.tau_mag_hi   # e.g. 99% training bound (Pa)

                mag = np.sqrt(tau_used[:, 0]**2 + tau_used[:, 1]**2)
                scale = np.minimum(1.0, cap / (mag + 1e-16))

                tau_used[:, 0] *= scale
                tau_used[:, 1] *= scale

                # Overwrite the field that the diagnostic solve will actually see
                self.tau_ml.dat.data[:] = tau_used

                if verbose:
                    # also report applied mean magnitude (useful sanity check)
                    mag_used_mean = float(np.mean(np.sqrt(tau_used[:, 0] ** 2 + tau_used[:, 1] ** 2)))
                    print(
                        f"[ML update {k+1}/{n_updates}] alpha={alpha:.3f} "
                        f"tau_pred mean=({tx_mean:.6g},{ty_mean:.6g}) Pa, "
                        f"std=({tx_std:.6g},{ty_std:.6g}) Pa, "
                        f"applied mean|tau|={mag_used_mean:.3g} Pa, "
                        f"max Δ|tau|={d_tau_max:.3g} Pa"
                    )

                if prev_tau is not None and d_tau_max < tol_tau:
                    if verbose:
                        print("   Converged on applied tau_ml field updates.")
                    break

                prev_tau = tau_used.copy()

            # --- diagnostic solve using current (fixed) applied tau_ml field ---
            u = self.diagnostic_solve(u, h, s, A, C, b)

        return u

    def compute_basal_total_drag_hybrid(
        self,
        u,
        h,
        s,
        b,
        A,
        *,
        qdeg=6,
        eps=1e-12,
        n=None,
        strain_rate_min=None,
    ):
        """
        Hybrid-consistent basal traction on the *graph bed* z=b(x,y), averaged over self.indicator on ds_b.

        Returns (same format as compute_form_drag_bp):
            bed_alongflow_stress : float (Pa)
            bed_vec_stress       : (float, float) (Pa, Pa)

        Sign convention:
        - We enforce Icepack's basal-drag convention (traction on the ice is dissipative):
                ∫_{ds_b} (t_xy · u) dA <= 0
            If the assembled basal power is positive, we flip (tx,ty) so that power is negative.
        """
        import firedrake
        import numpy as np
        import ufl
        from icepack.constants import glen_flow_law, strain_rate_min as _srmin

        if self.indicator is None:
            self.compute_box_indicator()

        if n is None:
            n = float(glen_flow_law)
        if strain_rate_min is None:
            strain_rate_min = float(_srmin)

        mesh = u.function_space().mesh()
        dim = mesh.geometric_dimension()
        ζ = firedrake.SpatialCoordinate(mesh)[dim - 1]

        # --- horizontal gradients only (x,y) on extruded mesh ---
        gb3 = firedrake.grad(b)
        gs3 = firedrake.grad(s)
        gb = firedrake.as_vector([gb3[0], gb3[1]])
        gs = firedrake.as_vector([gs3[0], gs3[1]])

        # --- Hybrid terrain-following horizontal strain rate (see hybrid.py) ---
        v_tf = -((1.0 - ζ) * gb + ζ * gs) / h          # 2-vector
        du_dζ = u.dx(dim - 1)                           # 2-vector

        Gu = firedrake.grad(u)                          # shape (2, dim)
        # take horizontal block
        Gu_xy = ufl.as_tensor([[Gu[0, 0], Gu[0, 1]],
                            [Gu[1, 0], Gu[1, 1]]])
        eps_sym = 0.5 * (Gu_xy + ufl.transpose(Gu_xy))

        corr = 0.5 * (ufl.outer(du_dζ, v_tf) + ufl.outer(v_tf, du_dζ))
        eps_x = eps_sym + corr                           # 2x2

        # --- Hybrid vertical shear strain rate ---
        eps_z = 0.5 * du_dζ / h                          # 2-vector

        # --- effective strain rate (hybrid.py _effective_strain_rate) ---
        eps_min = firedrake.Constant(strain_rate_min)
        ee = firedrake.sqrt(
            (firedrake.inner(eps_x, eps_x) + ufl.tr(eps_x) ** 2 + 2.0 * firedrake.inner(eps_z, eps_z)) / 2.0
            + eps_min**2
        )

        # --- stresses (hybrid.py stresses) ---
        mu = 0.5 * A ** (-1.0 / n) * ee ** (1.0 / n - 1.0)
        I2 = ufl.Identity(2)
        M = 2.0 * mu * (eps_x + ufl.tr(eps_x) * I2)      # 2x2 membrane stress
        tau_z = 2.0 * mu * eps_z                         # 2-vector vertical shear stress

        # --- hydrostatic pressure part (for normal-stress-on-slope contribution) ---
        X = firedrake.SpatialCoordinate(mesh)
        z = X[dim - 1]
        p = ρ_I * g * (s - z)

        # --- embed into 3x3 Cauchy stress tensor ---
        sigma = ufl.as_tensor([
            [M[0, 0] - p, M[0, 1],      tau_z[0]],
            [M[1, 0],     M[1, 1] - p,  tau_z[1]],
            [tau_z[0],    tau_z[1],     -p],
        ])

        # --- graph-bed normal from grad(b) ---
        bx = gb[0]
        by = gb[1]
        denom_n = firedrake.sqrt(1.0 + bx*bx + by*by)
        n_graph = firedrake.as_vector([-bx/denom_n, -by/denom_n, 1.0/denom_n])

        # traction on graph bed
        tvec = firedrake.dot(sigma, n_graph)
        tx = tvec[0]
        ty = tvec[1]

        dsb = firedrake.ds_b(metadata={"quadrature_degree": int(qdeg)})

        # --- enforce dissipative sign convention: ∫ (t_xy · u) dA <= 0 ---
        power = firedrake.assemble((tx * u[0] + ty * u[1]) * self.indicator * dsb)
        if float(power) > 0.0:
            tx = -tx
            ty = -ty

        # area over indicator on basal surface
        area = firedrake.assemble(self.indicator * dsb)
        area_val = float(area)
        if area_val <= 0.0:
            raise ValueError("Indicator area on ds_b is zero; check bbox/indicator.")

        tx_mean = float(firedrake.assemble(tx * self.indicator * dsb)) / area_val
        ty_mean = float(firedrake.assemble(ty * self.indicator * dsb)) / area_val

        tau_mag = float(np.sqrt(tx_mean * tx_mean + ty_mean * ty_mean))

        # along-flow projection using local flow direction
        u_mag = firedrake.max_value(firedrake.sqrt(u[0]**2 + u[1]**2), eps)
        flow_hat = firedrake.as_vector([u[0] / u_mag, u[1] / u_mag])

        txy = firedrake.as_vector([tx, ty])
        along_field = firedrake.dot(txy, flow_hat)
        along_int = firedrake.assemble(along_field * self.indicator * dsb)
        bed_alongflow_stress = float(along_int) / area_val

        print(
            f"[basal_total_drag_hybrid(graph)] area(ds_b)={area_val:.3e} m^2 | "
            f"tau=({tx_mean:.3e}, {ty_mean:.3e}) Pa | |tau|={tau_mag:.3e} Pa | "
            f"tau_along={bed_alongflow_stress:.3e} Pa | "
            f"power={float(power):.3e} (flipped={'yes' if float(power) > 0.0 else 'no'})"
        )

        return bed_alongflow_stress, (tx_mean, ty_mean)

    def compute_form_drag_proxy_bp(self, u, s, b):
        """
        Basal form-drag proxy (pressure-only, small-slope):
        tau_form(x,y) ≈ -p_b(x,y) ∇b(x,y)

        Returns (area-mean over indicator region on basal surface):
        bed_alongflow_stress : float (Pa)
        bed_vec_stress       : (float, float) (Pa, Pa)
        """
        eps = 1e-12

        if self.indicator is None:
            self.compute_box_indicator()

        # hydrostatic bed pressure (Pa)
        p_b = ρ_I * g * (s - b)

        # form-drag traction proxy (Pa): -p_b * ∇b
        bed_vec_field = -p_b * firedrake.grad(b)

        # basal area of the indicator region
        area = firedrake.assemble(self.indicator * firedrake.ds_b)
        area_val = float(area)
        if area_val <= 0.0:
            raise ValueError(
                "Indicator area on ds_b is zero. "
                "Check bbox bounds and indicator definition on the extruded mesh."
            )

        # integrate traction components over basal surface and area-average -> Pa
        bed_x = firedrake.assemble(bed_vec_field[0] * self.indicator * firedrake.ds_b)
        bed_y = firedrake.assemble(bed_vec_field[1] * self.indicator * firedrake.ds_b)

        tau_x = float(bed_x) / area_val
        tau_y = float(bed_y) / area_val
        tau_mag = (tau_x * tau_x + tau_y * tau_y) ** 0.5

        # along-flow (using local flow direction)
        u_mag = firedrake.max_value(firedrake.sqrt(u[0]**2 + u[1]**2), eps)
        flow_hat = firedrake.as_vector([u[0]/u_mag, u[1]/u_mag])

        bed_vec_field_xy = firedrake.as_vector([bed_vec_field[0], bed_vec_field[1]])
        bed_alongflow_field = firedrake.dot(bed_vec_field_xy, flow_hat)
        bed_alongflow = firedrake.assemble(bed_alongflow_field * self.indicator * firedrake.ds_b)
        bed_alongflow_stress = float(bed_alongflow) / area_val

        # --- debug magnitude print ---
        print(
            f"[form_drag_bp] area(ds_b)={area_val:.3e} m^2 | "
            f"tau=({tau_x:.3e}, {tau_y:.3e}) Pa | |tau|={tau_mag:.3e} Pa | "
            f"tau_along={bed_alongflow_stress:.3e} Pa"
        )

        return bed_alongflow_stress, (tau_x, tau_y)

    def compute_box_indicator(self):
        """
        Create an indicator function for the original box domain.
        :return: Indicator function (firedrake.Function)
        """
        xmin, xmax, ymin, ymax = self.get_original_bbox(self.padded_topo)
        mesh = self.mesh
        x, y, ζ = firedrake.SpatialCoordinate(mesh)

        # Indicator function for the box
        self.indicator = conditional(
            And(
                And(ge(x, xmin), le(x, xmax)),
                And(ge(y, ymin), le(y, ymax))
            ),
            1, 0
        )

    

    # def _box_mean(self, f):
    #     """Area-mean of scalar field f over indicator region."""
    #     if self.indicator is None:
    #         self.compute_box_indicator()
    #     area = assemble(self.indicator * firedrake.ds_b)
    #     return float(assemble(f * self.indicator * firedrake.ds_b) / area)

    # def _box_mean_vec(self, v):
    #     """Area-mean of 2D vector field v over indicator region."""
    #     if self.indicator is None:
    #         self.compute_box_indicator()
    #     area = assemble(self.indicator * firedrake.ds_b)
    #     vx = float(assemble(v[0] * self.indicator * firedrake.ds_b) / area)
    #     vy = float(assemble(v[1] * self.indicator * firedrake.ds_b) / area)
    #     return vx, vy

    # def debug_velocity_sign_and_bc(
    #     self,
    #     u,
    #     *,
    #     h=None,
    #     s=None,
    #     boundary_ids=(1, 2, 3, 4),
    #     indicator=None,
    #     indicator_threshold=0.5,
    #     print_pointwise_stats=True,
    # ):
    #     """
    #     Diagnostic suite to explain negative mean ux in a rectangular flow setup.

    #     Works with both 2D and extruded meshes (SpatialCoordinate may be (x,y) or (x,y,zeta)).
    #     """
    #     import numpy as np
    #     import firedrake

    #     results = {}

    #     if h is None:
    #         h = getattr(self, "h0", None)
    #     if s is None:
    #         s = getattr(self, "s0", None)
    #     if indicator is None:
    #         indicator = getattr(self, "indicator", None)

    #     # --- Mesh coordinate extents ---
    #     coords = self.mesh.coordinates.dat.data_ro
    #     x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    #     y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    #     results["x_min"] = x_min
    #     results["x_max"] = x_max
    #     results["y_min"] = y_min
    #     results["y_max"] = y_max
    #     print(f"[mesh] x in [{x_min:.3f}, {x_max:.3f}], y in [{y_min:.3f}, {y_max:.3f}]")

    #     # --- Boundary mean ux/uy ---
    #     def bnd_mean(expr, bid):
    #         den = firedrake.assemble(1.0 * firedrake.ds(bid))
    #         if float(den) == 0.0:
    #             return np.nan
    #         num = firedrake.assemble(expr * firedrake.ds(bid))
    #         return float(num / den)

    #     bnd = {}
    #     for bid in boundary_ids:
    #         try:
    #             ux_b = bnd_mean(u[0], bid)
    #             uy_b = bnd_mean(u[1], bid)
    #             bnd[bid] = (ux_b, uy_b)
    #             print(f"[boundary ds({bid})] mean(ux)={ux_b:.6g}, mean(uy)={uy_b:.6g}")
    #         except Exception as e:
    #             bnd[bid] = (np.nan, np.nan)
    #             print(f"[boundary ds({bid})] ERROR: {e}")
    #     results["boundary_means"] = bnd

    #     # --- Driving slope sign: mean(ds/dx) ---
    #     if s is not None:
    #         X = firedrake.SpatialCoordinate(self.mesh)
    #         x = X[0]  # works for 2D or extruded
    #         try:
    #             dsdx_mean = float(
    #                 firedrake.assemble(firedrake.diff(s, x) * firedrake.dx)
    #                 / firedrake.assemble(1.0 * firedrake.dx)
    #             )
    #             results["dsdx_mean"] = dsdx_mean
    #             print(f"[surface] mean(ds/dx) = {dsdx_mean:.6g}  (driving ~ -ds/dx)")
    #         except Exception as e:
    #             results["dsdx_mean"] = np.nan
    #             print(f"[surface] ERROR computing mean(ds/dx): {e}")
    #     else:
    #         results["dsdx_mean"] = np.nan
    #         print("[surface] s is None; skipping ds/dx diagnostic")

    #     # --- Thickness sanity ---
    #     if h is not None:
    #         try:
    #             # for extruded Function, dat.data_ro is still fine
    #             hvals = h.dat.data_ro
    #             results["h_min"] = float(np.min(hvals))
    #             results["h_max"] = float(np.max(hvals))
    #             print(f"[thickness] min(h)={results['h_min']:.6g}, max(h)={results['h_max']:.6g}")
    #         except Exception as e:
    #             print(f"[thickness] ERROR reading h range: {e}")
    #     else:
    #         print("[thickness] h is None; skipping thickness range")

    #     # --- Indicator / box diagnostics ---
    #     if indicator is not None:
    #         try:
    #             A = float(firedrake.assemble(indicator * firedrake.dx))
    #             results["indicator_area"] = A
    #             print(f"[indicator] area = {A:.6g}")
    #         except Exception as e:
    #             results["indicator_area"] = np.nan
    #             print(f"[indicator] ERROR computing area: {e}")

    #         # Area-mean ux in indicator
    #         try:
    #             denA = firedrake.assemble(indicator * firedrake.dx)
    #             if float(denA) != 0.0:
    #                 ux_area = float(firedrake.assemble(indicator * u[0] * firedrake.dx) / denA)
    #                 results["ux_area_mean_indicator"] = ux_area
    #                 print(f"[indicator] area-mean ux = {ux_area:.6g}")
    #             else:
    #                 results["ux_area_mean_indicator"] = np.nan
    #         except Exception as e:
    #             results["ux_area_mean_indicator"] = np.nan
    #             print(f"[indicator] ERROR computing area-mean ux: {e}")

    #         # Thickness-weighted mean u in indicator (matches training feature)
    #         if h is not None:
    #             try:
    #                 denH = firedrake.assemble(h * indicator * firedrake.dx)
    #                 if float(denH) != 0.0:
    #                     ux_h = float(firedrake.assemble(h * indicator * u[0] * firedrake.dx) / denH)
    #                     uy_h = float(firedrake.assemble(h * indicator * u[1] * firedrake.dx) / denH)
    #                     results["ux_h_mean_indicator"] = ux_h
    #                     results["uy_h_mean_indicator"] = uy_h
    #                     print(f"[indicator] h-weighted mean (ux,uy)=({ux_h:.6g},{uy_h:.6g})")
    #                 else:
    #                     results["ux_h_mean_indicator"] = np.nan
    #                     results["uy_h_mean_indicator"] = np.nan
    #                     print("[indicator] WARNING: ∫h*I dx == 0; cannot compute h-weighted mean u")
    #             except Exception as e:
    #                 results["ux_h_mean_indicator"] = np.nan
    #                 results["uy_h_mean_indicator"] = np.nan
    #                 print(f"[indicator] ERROR computing h-weighted mean u: {e}")

    #     else:
    #         print("[indicator] indicator is None; skipping indicator diagnostics")

    #     # --- Pointwise min/max ux diagnostics ---
    #     if print_pointwise_stats:
    #         try:
    #             # Use scalar space for projection; prefer self.Q if available (likely 2D base space)
    #             Q = getattr(self, "Q", None)
    #             if Q is None:
    #                 Q = firedrake.FunctionSpace(self.mesh, "CG", 1)
    #             ux_proj = firedrake.project(u[0], Q)
    #             ux_vals = ux_proj.dat.data_ro
    #             results["ux_min_domain"] = float(np.min(ux_vals))
    #             results["ux_max_domain"] = float(np.max(ux_vals))
    #             print(f"[ux] domain min={results['ux_min_domain']:.6g}, max={results['ux_max_domain']:.6g}")

    #             if indicator is not None:
    #                 I_proj = firedrake.project(indicator, Q)
    #                 mask = I_proj.dat.data_ro > float(indicator_threshold)
    #                 if np.any(mask):
    #                     results["ux_min_indicator"] = float(np.min(ux_vals[mask]))
    #                     results["ux_max_indicator"] = float(np.max(ux_vals[mask]))
    #                     print(f"[ux] indicator min={results['ux_min_indicator']:.6g}, max={results['ux_max_indicator']:.6g}")
    #                 else:
    #                     print("[ux] indicator mask empty (after projection/threshold)")
    #         except Exception as e:
    #             print(f"[ux] ERROR computing pointwise stats: {e}")

    #     print("[debug] done.")
    #     return results


    def load_models_and_percentiles(
            self,
            encoder_path="vae/encoder_model",
            decoder_path="vae/decoder_model",
            percentiles_path="vae/latent_percentiles.pkl",
            call_endpoint="serving_default",
            encoder_input_shape=(None, None, 1),  # only matters if you actually use encoder.predict
        ):
        """
        Load encoder, decoder models and latent percentiles from disk.

        Keras 3 note:
        - load_model() does NOT load SavedModel directories.
        - We use TFSMLayer to wrap SavedModel for inference.
        """

        # Percentiles
        with open(percentiles_path, "rb") as f:
            percentiles = pickle.load(f)

        # Support both dict style {'lower':..., 'upper':...} and tuple style (lower, upper)
        if isinstance(percentiles, dict) and "lower" in percentiles and "upper" in percentiles:
            lower, upper = percentiles["lower"], percentiles["upper"]
        else:
            lower, upper = percentiles

        lower = np.asarray(lower)
        upper = np.asarray(upper)
        latent_dim = int(lower.shape[0])

        def _normalize_output(y):
            # TFSMLayer may return Tensor, dict, list/tuple
            if isinstance(y, dict):
                return y[sorted(y.keys())[0]]
            if isinstance(y, (list, tuple)):
                return y[0]
            return y

        # Decoder model (z -> image)
        decoder_layer = keras.layers.TFSMLayer(decoder_path, call_endpoint=call_endpoint)
        z_in = keras.Input(shape=(latent_dim,), name="z")
        dec_out = _normalize_output(decoder_layer(z_in))
        decoder = keras.Model(z_in, dec_out, name="decoder_infer")

        # Encoder model (image -> z), optional but kept for API compatibility
        encoder = None
        try:
            encoder_layer = keras.layers.TFSMLayer(encoder_path, call_endpoint=call_endpoint)
            x_in = keras.Input(shape=encoder_input_shape, name="x")
            enc_out = _normalize_output(encoder_layer(x_in))
            encoder = keras.Model(x_in, enc_out, name="encoder_infer")
        except Exception as e:
            # If you don't use encoder anywhere, it's fine to leave as None.
            print(f"Warning: could not wrap encoder SavedModel ({e}). Returning encoder=None.")

        print(
            f"Loaded encoder (TFSMLayer) from {encoder_path}, "
            f"decoder (TFSMLayer) from {decoder_path}, "
            f"percentiles from {percentiles_path}"
        )
        return encoder, decoder, lower, upper

    
        # ---------- ML attachment / loading ----------

    # def attach_ml_drag_model(self, model, scaler_input, scaler_output, feature_names):
    #     """
    #     Attach an already-loaded XGBoost model + scalers.
    #     feature_names must match the training column order exactly.
    #     """
    #     self.ml_model = model
    #     self.ml_scaler_input = scaler_input
    #     self.ml_scaler_output = scaler_output
    #     self.ml_feature_names = list(feature_names)

    #     # These are run-time state holders
    #     #self.tau_ml_const = Constant(0.0)  # Pa
    #     self.use_ml_drag = True

    def create_topography(
        self,
        encoder_path="vae/encoder_model",
        decoder_path="vae/decoder_model",
        percentiles_path="vae/latent_percentiles.pkl",
        z_samples=None,
        n=1,
    ):
        """
        Create topography using a trained VAE decoder.

        Parameters
        ----------
        z_samples : array-like or None
            Latent vectors to decode.
            - If None: sample uniformly between stored latent percentiles.
            - If shape (latent_dim,): treated as a single sample.
            - If shape (n, latent_dim): treated as a batch.
        n : int
            Number of samples to generate when z_samples is None.
        encoder_path, decoder_path, percentiles_path : str
            Paths for loading encoder/decoder and latent percentiles.

        Returns
        -------
        generated : np.ndarray
            Decoded topography image(s). If one sample, returns a single image array.
            If batch, returns an array of images.
        z_used : np.ndarray
            The latent vector(s) actually used (shape (n, latent_dim)).
        """
        # Load models and percentiles
        encoder_loaded, decoder_loaded, lower, upper = self.load_models_and_percentiles(
            encoder_path, decoder_path, percentiles_path
        )

        lower = np.asarray(lower)
        upper = np.asarray(upper)
        latent_dim = lower.shape[0]

        # Decide latent vectors to use
        if z_samples is None:
            if n is None or n < 1:
                raise ValueError("When z_samples is None, n must be an integer >= 1.")
            z_used = np.random.uniform(low=lower, high=upper, size=(int(n), latent_dim))
        else:
            z_used = np.asarray(z_samples, dtype=np.float32)
            if z_used.ndim == 1:
                if z_used.shape[0] != latent_dim:
                    raise ValueError(f"z_samples has length {z_used.shape[0]} but expected {latent_dim}.")
                z_used = z_used[None, :]  # (1, latent_dim)
            elif z_used.ndim == 2:
                if z_used.shape[1] != latent_dim:
                    raise ValueError(f"z_samples has dim {z_used.shape[1]} but expected {latent_dim}.")
            else:
                raise ValueError("z_samples must be None, shape (latent_dim,), or shape (n, latent_dim).")

        # Decode
        generated = decoder_loaded.predict(z_used)

        # Match your old return style: single image if only one sample
        if generated.shape[0] == 1:
            print("latent vector used:", z_used)
            return generated[0], z_used
        return generated, z_used

    def create_topography_scaled(self, encoder_path='vae/encoder_model', decoder_path='vae/decoder_model', percentiles_path='vae/latent_percentiles.pkl', multiplier=1, z_samples=None):
        # b = np.load('thw_image_0.npy') # temp, use model
        b, latent_vector = self.create_topography(encoder_path, decoder_path, percentiles_path, z_samples=z_samples)
        return b[:,:,0] * multiplier , latent_vector
    
    def pad_topography(self, data, pad_x_minus, pad_x_plus, pad_y, mode = 'reflect'):
        """
        Pads the input data with specified padding methods.
        Parameters:
        data (numpy.ndarray): The input data to be padded.
        pad_x_minus (int): Padding size on the left side of the data.
        pad_x_plus (int): Padding size on the right side of the data.
        pad_y (int): Padding size on the top and bottom of the data.
        mode (str): Padding mode. Default is 'reflect'.
                    Other options include 'constant', 'edge', 'linear_ramp', etc.
        Returns:
        numpy.ndarray: Padded data.
        """
        self.pad_x_minus = pad_x_minus
        self.pad_x_plus = pad_x_plus
        self.pad_y = pad_y
        padded_data = np.pad(
            data,
            pad_width=((self.pad_y, self.pad_y), (self.pad_x_minus, self.pad_x_plus)),
            mode=mode
        )
        
        return padded_data
    
    def save_tif(self, padded_data, pixel_size_x, pixel_size_y, filename="output.tif"):
        """
        Save the padded data as a GeoTIFF file.
        :param padded_data: Padded data to be saved.
        :param pixel_size_x: Pixel size in the x-direction.
        :param pixel_size_y: Pixel size in the y-direction.
        :param filename: Path to the output GeoTIFF file.
        :return: Affine transform object.
        """
        height, width = padded_data.shape
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        x_origin = 0
        y_origin = padded_data.shape[0] * self.pixel_size_y
        # Shift so pixel centers start at (-25, -25)
        transform = Affine(self.pixel_size_x, 0.0, x_origin,
                    0.0, -self.pixel_size_y, y_origin) # Note negative pixel size for Y

        with rasterio.open(
            filename,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=padded_data.dtype,
            crs='EPSG:3413', # Correct CRS
            transform=transform
        ) as dst:
            dst.write(padded_data, 1)

        return transform
    
    def get_original_bbox(self, padded_data):
        """
        Get bounding box of the original (unpadded) data in projected coordinates.
        Returns (xmin, xmax, ymin, ymax).
        """
        height, width = padded_data.shape
        
        # Full padded extents
        xmin_padded = 0
        xmax_padded = width * self.pixel_size_x
        ymin_padded = 0
        ymax_padded = height * self.pixel_size_y
        
        # Trim padding in physical units
        xmin = xmin_padded + self.pad_x_minus * self.pixel_size_x
        xmax = xmax_padded - self.pad_x_plus * self.pixel_size_x
        ymin = ymin_padded + self.pad_y * self.pixel_size_y
        ymax = ymax_padded - self.pad_y * self.pixel_size_y
        
        return xmin, xmax, ymin, ymax
    
    def plot_padded_topography(self, padded_topo):
        # Compute physical extents of the full padded image
        height, width = padded_topo.shape
        extent = [0, width * self.pixel_size_x, 0, height * self.pixel_size_y]
        
        # Plot the padded topo
        plt.imshow(padded_topo, extent=extent, origin="lower")
        plt.title('Subglacial topography data')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        color_bar = plt.colorbar()
        color_bar.set_label('Padded Bed Elevation (m)')
        
        # Overlay rectangle of original bbox
        xmin, xmax, ymin, ymax = self.get_original_bbox(padded_topo)
        rect = plt.Rectangle(
            (xmin, ymin),                # lower-left corner
            xmax - xmin,                 # width
            ymax - ymin,                 # height
            linewidth=2, edgecolor='red', facecolor='none'
        )
        plt.gca().add_patch(rect)
        
        plt.show()
    
    def create_geojson_bbox_from_tif(self, tif_path, geojson_path):
        """
        Create a GeoJSON file with 4 lines representing the bounding box of a GeoTIFF file.
        :param tif_path: Path to the input GeoTIFF file.
        :param geojson_path: Path to the output GeoJSON file.
        """
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            transform = src.transform
            crs = src.crs

            # Get pixel size
            pixel_width = transform.a
            pixel_height = -transform.e  # Usually negative in affine transform

            factor = 2
            # Offset bounds by 1 pixel inward
            left = bounds.left + factor*pixel_width
            right = bounds.right - factor*pixel_width
            bottom = bounds.bottom + factor*pixel_height
            top = bounds.top - factor*pixel_height

            # Define 4 lines (edges of the bounding box), 1 pixel inside
            lines = [
                LineString([(left, bottom), (right, bottom)]),  # bottom
                LineString([(right, bottom), (right, top)]),    # right
                LineString([(right, top), (left, top)]),        # top
                LineString([(left, top), (left, bottom)])       # left
            ]

            # Convert lines to GeoJSON features
            features = [
                {
                    "type": "Feature",
                    "geometry": mapping(line),
                    "properties": {"edge": name}
                }
                for line, name in zip(lines, ["bottom", "right", "top", "left"])
            ]

            # Assemble FeatureCollection
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }

            with open(geojson_path, 'w') as f:
                json.dump(geojson, f)

            print(f"GeoJSON with 4 bounding box lines written to {geojson_path}")

    def create_processed_topography(self, encoder_path=None, decoder_path=None, percentiles_path=None, index=None, scaling_multiplier=500, pad_x_minus=100, pad_x_plus=100, pad_y=10, pixel_size_x=25, pixel_size_y=25, z_samples=None, create_constant = False):
        """
        Create processed topography using a trained model.
        :param model_filename: Path to the trained model file.
        :param pad_x_minus: Padding size on the left side of the data.
        :param pad_x_plus: Padding size on the right side of the data.
        :param pad_y: Padding size on the top and bottom of the data.
        :param pixel_size_x: Pixel size in the x-direction.
        :param pixel_size_y: Pixel size in the y-direction.
        :param create_constant: Whether to create a constant topography.
        :return: Generated images.
        """
        if create_constant:
            b = np.ones((512, 512)) * scaling_multiplier  # Example constant topography
            latent_vector = None
        else:
            b, latent_vector = self.create_topography_scaled(encoder_path=encoder_path, decoder_path=decoder_path, percentiles_path=percentiles_path, multiplier=scaling_multiplier, z_samples=z_samples)
        self.padded_topo = self.pad_topography(b, pad_x_minus, pad_x_plus, pad_y)
        if not index:
            self.random_index = np.random.randint(0, 100000)
        else:
            self.random_index = index
        self.filename = f"generated_data/base_{self.random_index}.tif"
        self.geo_filename = f"generated_data/boundary_{self.random_index}.geojson"
        transform = self.save_tif(self.padded_topo, pixel_size_x, pixel_size_y, filename=self.filename)
        self.create_geojson_bbox_from_tif(self.filename, self.geo_filename)
        self.padded_topo = self.pad_topography(b, pad_x_minus+1, pad_x_plus+1, pad_y+1)
        transform = self.save_tif(self.padded_topo, pixel_size_x, pixel_size_y, filename=self.filename)

        return self.padded_topo, transform, latent_vector

    def create_mesh_synthetic(self, filename = None, lcar = 1e3):
        """
        Create a mesh using the provided filename or a default one.
        :param filename: Path to the GeoJSON file.
        :param lcar: Mesh size parameter.
        """
        if not filename:
            filename = self.geo_filename
        outline = fetch_outline(filename)
        create_mesh(outline, name=f"mesh_{self.random_index}", lcar=lcar)
        self.mesh2d = Mesh(f"mesh_{self.random_index}.msh")
        self.mesh = firedrake.ExtrudedMesh(self.mesh2d, layers=1)

    def create_rectangle_mesh_synthetic(self, filename = None, nx = 48, ny = 32):
        """
        Create a mesh using the provided filename or a default one.
        :param filename: Path to the GeoJSON file.
        :param lcar: Mesh size parameter.
        """
        if not filename:
            filename = self.geo_filename

        outline = fetch_outline(filename)
        if outline is None:
            raise RuntimeError("Could not load outline.")

        self.Lx, self.Ly, self.originX, self.originY = get_bbox_from_outline(outline)

        self.mesh2d = create_rectangle_mesh(nx, ny, self.Lx, self.Ly, originX=self.originX, originY=self.originY)
        self.mesh = firedrake.ExtrudedMesh(self.mesh2d, layers=1)

    def create_function_space(self):
        """
        Create function spaces for the model.
        """
        self.Q = firedrake.FunctionSpace(
            self.mesh, "CG", 2, vfamily="R", vdegree=0
        )
        self.V = firedrake.VectorFunctionSpace(
            self.mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2
        )

        # Scalar space aligned with velocity DOFs (same mesh + discretization family as each component of V)
        #self.V0 = self.V.sub(0).collapse()

        #self.tau_ml_const = firedrake.Constant((0.0, 0.0))
        #self.tau_ml = firedrake.Function(self.V, name="tau_ml")  # (Pa, Pa) at DOFs
        #self.tau_ml.assign(firedrake.Constant((0.0, 0.0)))

        self.Q_dg0 = firedrake.FunctionSpace(self.mesh, "DG", 0)
        self.V_dg0 = firedrake.VectorFunctionSpace(self.mesh, "DG", 0, dim=2)

        self.tau_ml = firedrake.Function(self.V_dg0, name="tau_ml")  # DG0 traction (Pa)
        self.tau_ml.assign(firedrake.Constant((0.0, 0.0)))
    
    def set_initial_conditions(self, uniform_thickness = 500, surface_slope = -200, u_in = 20, u_out = 200, constant_temperature = 260, constant_C=1, m = 3.0):
        """
        Set initial conditions for the model.
        :param uniform_thickness: Uniform thickness of the ice.
        :param surface_slope: Surface slope of the ice. The net change in surface height over the domain length in x-direction.
        :param u_in: Inlet velocity.
        :param u_out: Outlet velocity.
        :param constant_temperature: Constant temperature.
        :param constant_C: Constant C value.
        :param m: Glen's flow law exponent.
        """
        x, y, ζ = firedrake.SpatialCoordinate(self.mesh)

        self.m = firedrake.Constant(m)

        b_expr = rasterio.open(self.filename)
        self.b = icepack.interpolate(b_expr, self.Q)

        if uniform_thickness is not None and surface_slope is None:
            print(f"Using uniform thickness: {uniform_thickness} m")
            s_expr = self.b + uniform_thickness

        elif uniform_thickness is not None and surface_slope is not None:
            print(f"Using starting thickness: {uniform_thickness} m")
            print(f"Using surface slope: {surface_slope}")
            s_expr = uniform_thickness + (x * surface_slope / self.Lx)
            self.real_surface_slope = surface_slope / self.Lx
            print(f"Real surface slope (m/m): {self.real_surface_slope}")

        else:
            raise ValueError("You must provide either uniform_thickness, or both uniform_thickness and surface_slope.")
        
        self.s0 = firedrake.interpolate(s_expr, self.Q)

        h_expr = self.s0 - self.b
        self.h0 = firedrake.interpolate(h_expr, self.Q)

        if u_out is None:
            u_out = u_in * 10

        velocity_x = u_in + (u_out - u_in) * (x/self.Lx)**2
        self.u0 = firedrake.interpolate(firedrake.as_vector((velocity_x, 0)), self.V)

        T = firedrake.Constant(constant_temperature)
        self.A = icepack.rate_factor(T)

        # expr = (ρ_I * g * self.h0 * firedrake.sqrt(inner(grad(self.s0), grad(self.s0))) / (firedrake.sqrt(inner(self.u0, self.u0)) ** (1/self.m)))
        # self.C = icepack.interpolate(expr, self.Q) 
        self.C = firedrake.Constant(constant_C)

        self.compute_box_indicator()

    # def friction(self, **kwargs):
    #     """
    #     Friction model for the hybrid model.
    #     :param kwargs: Keyword arguments containing velocity, thickness, surface, and friction.
    #     :return: Friction term.
    #     """
    #     u = kwargs["velocity"]
    #     h = kwargs["thickness"]
    #     s = kwargs["surface"]
    #     C = kwargs["friction"]

    #     p_W = ρ_W * g * firedrake.max_value(0, h - s)
    #     p_I = ρ_I * g * h
    #     ϕ = 1 - p_W / p_I
    #     return icepack.models.hybrid.bed_friction(
    #         velocity=u,
    #         friction=C * ϕ,
    #     )

    def constant_friction(self, **kwargs):
        """
        Total basal traction:
        tau_total = -C * u/|u|  +  tau_ml
        Returns power density tau_total · u (Icepack convention).
        """
        u, C = kwargs["velocity"], kwargs["friction"]
        eps = firedrake.Constant(1e-12)
        r = firedrake.sqrt(firedrake.inner(u, u) + eps*eps)

        tau_c = -C * u / r

        use_ml = getattr(self, "use_ml_drag", False) and getattr(self, "tau_ml", None) is not None
        tau_total = tau_c + self.tau_ml if use_ml else tau_c

        # ---- Debug (mean |tau_total| over your indicator region) ----
        # WARNING: constant_friction is evaluated many times during Newton/Picard,
        # so guard prints or you'll spam.
        if use_ml and getattr(self, "debug_tau", False):
            if self.indicator is None:
                self.compute_box_indicator()

            area = firedrake.assemble(self.indicator * firedrake.dx)
            mean_mag = firedrake.assemble(firedrake.sqrt(firedrake.inner(tau_total, tau_total))
                                        * self.indicator * firedrake.dx) / area

            # optional: also mean |tau_ml|
            mean_mag_ml = firedrake.assemble(firedrake.sqrt(firedrake.inner(self.tau_ml, self.tau_ml))
                                            * self.indicator * firedrake.dx) / area

            # print only once per outer update if you want
            k = getattr(self, "_tau_print_k", 0)
            if getattr(self, "_tau_printed_k", None) != k:
                self._tau_printed_k = k
                print(f"[k={k}] mean |tau_total|={float(mean_mag):.3g} Pa ; mean |tau_ml|={float(mean_mag_ml):.3g} Pa")
            #print(f"............................ mean |tau_total|={float(mean_mag):.3g} Pa ; mean |tau_ml|={float(mean_mag_ml):.3g} Pa")


        return -firedrake.inner(tau_total, u)


    def terminus(self, **kwargs):
        """
        Modified terminus boundary condition for the hybrid model to allow for 
        both marine and land-terminating glaciers.

        The power exerted due to stress at the terminus is now given by:

        .. math::
            E(u) = \int_\Gamma\int_0^1\left(\rho_Ig(1 - \zeta) -
            \rho_Ig(\zeta_{\text{downstream}} - \zeta)_+\right)u\cdot\nu\; h\, d\zeta\; ds

        where :math:`\zeta_{\text{downstream}}` accounts for the downstream ice thickness
        instead of assuming a water pressure term.

        Parameters
        ----------
        u : firedrake.Function
            Ice velocity
        h : firedrake.Function
            Ice thickness
        s : firedrake.Function
            Ice surface elevation
        b : firedrake.Function
            Ice bed elevation
        """
        u, h, s, b = itemgetter("velocity", "thickness", "surface", "bed")(kwargs)

        mesh = u.function_space().mesh()
        zdegree = u.ufl_element().degree()[1]

        ζ = firedrake.SpatialCoordinate(mesh)[mesh.geometric_dimension() - 1]

        d = s - b  # Ice thickness downstream (no assumption of flotation)
        ζ_ds = d / h  # Relative depth to the downstream surface
        p_D = ρ_I * g * h * _pressure_approx(zdegree + 1)(ζ, ζ_ds)  # Downstream ice pressure
        p_I = ρ_I * g * h * (1 - ζ)  # Local ice pressure

        ν = FacetNormal(mesh)
        return (p_I - p_D) * inner(u, ν) * h

        
    def create_model(self, drichlet_ids=[4], side_wall_ids=[1, 3]):
        """
        Create the model for the simulation.
        :param drichlet_ids: List of Dirichlet boundary IDs.
        :param side_wall_ids: List of side wall boundary IDs.
        """
        model = icepack.models.HybridModel(friction=self.constant_friction, terminus=self.terminus)
        self.diagnostic_solve_model = model
        opts = {"dirichlet_ids": drichlet_ids, "side_wall_ids": side_wall_ids}
        self.solver = icepack.solvers.FlowSolver(self.diagnostic_solve_model, **opts)

    def diagnostic_solve(self, u0, h0, s0, A, C, b):
        """
        Diagnostic solve for the model.
        :param u0: Initial velocity.
        :param h0: Initial thickness.
        :param s0: Initial surface.
        :param A: Fluidity.
        :param C: Friction.
        :return: Diagnostic solution.
        """
        return self.solver.diagnostic_solve(
            velocity=u0,
            thickness=h0,
            surface=s0,
            fluidity=A,
            friction=C,
            bed = b,
        )
        
    def prognostic_solve(self, h0, u0, num_years=20, timesteps_per_year=10):
        """
        Prognostic solve for the model.
        :param h0: Initial thickness.
        :param u0: Initial velocity.
        :param num_years: Number of years for the simulation.
        :param timesteps_per_year: Number of timesteps per year.
        :return: Prognostic solution.
        """
        δt = 1.0 / timesteps_per_year
        num_timesteps = num_years * timesteps_per_year

        x, y, ζ = firedrake.SpatialCoordinate(self.mesh)
        a = firedrake.interpolate(1.7 - 2.7 * x / self.Lx, self.Q)
        h = h0.copy(deepcopy=True)
        u = u0.copy(deepcopy=True)

        for step in tqdm.trange(num_timesteps):
            h = self.solver.prognostic_solve(
                δt,
                thickness=h,
                velocity=u,
                accumulation=a,
                thickness_inflow=h0,
            )
            s = icepack.compute_surface(thickness=h, bed=self.b)
            
            u = self.diagnostic_solve(
                u,
                h,
                s,
                self.A,
                self.C,
                self.b,
            )
        return h, u, s

    def setup_model(self, filename = None, uniform_thickness = 500,surface_slope = -200, u_in = 20, u_out = 200, constant_temperature = 260, constant_C=1, drichlet_ids = [4], side_wall_ids = [1, 3], nx = 48, ny = 32):
        """
        Setup the model with the given parameters.
        :param filename: Path to the GeoJSON file.
        :param lcar: Mesh size parameter.
        :param uniform_thickness: Uniform thickness of the ice.
        :param u_in: Inlet velocity.
        :param u_out: Outlet velocity.
        :param constant_temperature: Constant temperature.
        :param constant_C: Constant C value.
        :param drichlet_ids: List of Dirichlet boundary IDs.
        :param side_wall_ids: List of side wall boundary IDs.
        """
        # self.create_mesh_synthetic(filename = filename, lcar = lcar)
        # self.create_function_space()
        # self.set_initial_conditions(uniform_thickness=uniform_thickness, u_in=u_in, u_out=u_out, constant_temperature=constant_temperature, constant_C=constant_C)
        # self.create_model(drichlet_ids = drichlet_ids, side_wall_ids = side_wall_ids)
        if filename is None:
            filename = self.geo_filename
            print(f"Using default filename: {filename}")
        self.create_rectangle_mesh_synthetic(filename = filename, nx = nx, ny = ny)
        self.create_function_space()
        self.set_initial_conditions(uniform_thickness=uniform_thickness, surface_slope=surface_slope, u_in=u_in, u_out=u_out, constant_temperature=constant_temperature, constant_C=constant_C)
        self.create_model(drichlet_ids=drichlet_ids, side_wall_ids=side_wall_ids)

    def create_synthetic_setup(self):
        # Implement the logic to create synthetic data
        pass

    def save_synthetic_data(self, filename):
        # Implement the logic to save synthetic data
        pass