import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import xgboost as xgb
import random
import pandas as pd
import matplotlib.pyplot as plt
import firedrake
from setuptools import setup
import icepack
from src.create_synthetic import CreateSynthetic
import concurrent.futures
import gc
import seaborn as sns
from sklearn.metrics import r2_score
import joblib


def plot_padded_topography(padded_topo):
    plt.imshow(padded_topo)
    plt.title('Subglacial topography data')
    plt.xlabel('X')
    plt.ylabel('Y')
    color_bar = plt.colorbar()
    color_bar.set_label('Padded Bed Elevation (m)')

def plot_high_res_mesh(create_synthetic):
    fig, axes = icepack.plot.subplots()
    firedrake.triplot(create_synthetic.mesh2d, axes=axes)
    axes.set_title("Mesh");
    plt.legend()

def plot_depth_average_u(u0):
    fig, axes = icepack.plot.subplots()
    colors = firedrake.tripcolor(icepack.depth_average(u0), axes=axes)
    fig.colorbar(colors, ax=axes, fraction=0.012, pad=0.04);

def run_synthetic_high_res(encoder_path=None, decoder_path=None, percentiles_path=None, 
                           index = None, 
                           scaling_multiplier = 500, 
                           pad_x_minus= 600, 
                           pad_x_plus = 200, 
                           pad_y = 10, 
                           pixel_size_x = 50, 
                           pixel_size_y = 50, 
                           filename = None, 
                           uniform_thickness=2000, 
                           surface_slope=-300, 
                           u_in = 200, 
                           u_out = None, 
                           constant_temperature = 260, 
                           constant_C=0.01, 
                           drichlet_ids = [1], 
                           side_wall_ids = [3, 4],
                           nx = 48, ny = 32,
                           plot_topography = False,
                           plot_mesh = False,
                           plot_depth_average_vel = False, 
                           z_samples=None,
                           create_constant=False
                           ):
    """
    Runs a high-resolution synthetic ice flow model.
    """
    create_synthetic = CreateSynthetic()
    padded_topo, transform, latent_vector = create_synthetic.create_processed_topography(encoder_path=encoder_path, decoder_path=decoder_path, percentiles_path=percentiles_path, index=index, scaling_multiplier=scaling_multiplier, pad_x_minus=pad_x_minus, pad_x_plus=pad_x_plus, pad_y=pad_y, pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y, z_samples=z_samples, create_constant=create_constant)
    if plot_topography:
        create_synthetic.plot_padded_topography(padded_topo)
    create_synthetic.setup_model(filename = filename, uniform_thickness=uniform_thickness, surface_slope=surface_slope, u_in = u_in, u_out = u_out, constant_temperature = constant_temperature, constant_C=constant_C, drichlet_ids = drichlet_ids, side_wall_ids = side_wall_ids, nx = nx, ny = ny)
    if plot_mesh:
        plot_high_res_mesh(create_synthetic)
    u = create_synthetic.diagnostic_solve(create_synthetic.u0, create_synthetic.h0, create_synthetic.s0, create_synthetic.A, create_synthetic.C, create_synthetic.b)
    if plot_depth_average_vel:
        plot_depth_average_u(u)
    return u, create_synthetic, latent_vector

# def run_synthetic_coarse_with_ml_drag(
#     encoder_path=None,
#     decoder_path=None,
#     percentiles_path=None,
#     index=None,
#     scaling_multiplier=500,
#     pad_x_minus=600,
#     pad_x_plus=200,
#     pad_y=10,
#     pixel_size_x=50,
#     pixel_size_y=50,
#     filename=None,
#     uniform_thickness=2000,
#     surface_slope=-300,            # only used to build initial s0 in your setup_model
#     u_in=200,
#     u_out=None,
#     constant_temperature=260,
#     constant_C=0.01,
#     drichlet_ids=[1],
#     side_wall_ids=[3, 4],
#     nx=12, ny=8,                  # <-- coarse mesh
#     plot_topography=False,
#     plot_mesh=False,
#     plot_depth_average_vel=False,
#     z_samples=None,
#     # --- ML / coupling controls ---
#     model_path="xgb_shear_model",
#     k_train_slope=None,            # <-- REQUIRED for slope equivalence
#     strip_frac=0.05,
#     n_updates=10,
#     tol_tau=1e-2,
#     use_indicator_mean=True,
# ):
#     """
#     Runs a COARSE synthetic ice flow model with ML form-drag coupling.

#     Steps:
#       1) generate synthetic bed + latent vector
#       2) setup coarse model, initial diagnostic solve without ML (optional but stable)
#       3) set ML constant features: slope_equiv, latent, roughness metrics
#       4) run outer fixed-point loop: update tau_ml_const -> diagnostic_solve -> repeat

#     Returns:
#       u_final, create_synthetic, latent_vector, (tau_x, tau_y)
#     """
#     if k_train_slope is None:
#         raise ValueError("k_train_slope is required (training-calibrated slope factor).")

#     create_synthetic = CreateSynthetic()

#     # --- generate bed/topo (same as high-res path) ---
#     padded_topo, transform, latent_vector = create_synthetic.create_processed_topography(
#         encoder_path=encoder_path,
#         decoder_path=decoder_path,
#         percentiles_path=percentiles_path,
#         index=index,
#         scaling_multiplier=scaling_multiplier,
#         pad_x_minus=pad_x_minus,
#         pad_x_plus=pad_x_plus,
#         pad_y=pad_y,
#         pixel_size_x=pixel_size_x,
#         pixel_size_y=pixel_size_y,
#         z_samples=z_samples,
#     )

#     if plot_topography:
#         create_synthetic.plot_padded_topography(padded_topo)

#     # --- setup COARSE mesh model ---
#     create_synthetic.setup_model(
#         filename=filename,
#         uniform_thickness=uniform_thickness,
#         surface_slope=surface_slope,
#         u_in=u_in,
#         u_out=u_out,
#         constant_temperature=constant_temperature,
#         constant_C=constant_C,
#         drichlet_ids=drichlet_ids,
#         side_wall_ids=side_wall_ids,
#         nx=nx,
#         ny=ny,
#     )

#     if plot_mesh:
#         plot_high_res_mesh(create_synthetic)  # re-uses your plotter; naming ok

#     # --- load ML drag model + enable ML drag in friction ---
#     create_synthetic.load_ml_drag_model(model_path=model_path)
#     create_synthetic.set_use_ml_drag(True)

#     # --- compute training-equivalent slope over *this* box from current surface ---
#     # box_surface_drop_x should use indicator or bbox; relies on your implementation
#     delta_s_box_current = create_synthetic.box_surface_drop_x(create_synthetic.s0, strip_frac=strip_frac)
#     surface_slope_equiv = float(delta_s_box_current) / float(k_train_slope)

#     # --- roughness metrics (for now: from current bed in the box) ---
#     rms_roughness, slope_roughness = create_synthetic.compute_roughness_metrics(create_synthetic.b)

#     # --- set constant ML features ONCE ---
#     create_synthetic.set_ml_constant_features(
#         surface_slope_equiv=surface_slope_equiv,
#         latent_vector=latent_vector[0] if np.asarray(latent_vector).ndim == 2 else latent_vector,
#         rms_roughness=rms_roughness,
#         slope_roughness=slope_roughness,
#     )

#     # --- optional: initial diagnostic solve WITHOUT ML to get a reasonable u to start from ---
#     # (You can skip this; often helps stability)
#     create_synthetic.set_use_ml_drag(False)
#     u0 = create_synthetic.diagnostic_solve(
#         create_synthetic.u0,
#         create_synthetic.h0,
#         create_synthetic.s0,
#         create_synthetic.A,
#         create_synthetic.C,
#         create_synthetic.b,
#     )
#     create_synthetic.set_use_ml_drag(True)

#     # --- coupled solve with ML updates ---
#     u_final, (tau_x, tau_y) = create_synthetic.coarse_diagnostic_solve_with_ml_updates(
#         u_init=u0,
#         h=create_synthetic.h0,
#         s=create_synthetic.s0,
#         A=create_synthetic.A,
#         C=create_synthetic.C,
#         b=create_synthetic.b,
#         temperature_K=float(constant_temperature),
#         C_value=float(constant_C),
#         n_updates=n_updates,
#         use_indicator_mean=use_indicator_mean,
#         tol_tau=tol_tau,
#         verbose=True,
#     )

#     if plot_depth_average_vel:
#         plot_depth_average_u(u_final)

#     return u_final, create_synthetic, latent_vector, (tau_x, tau_y)


def run_synthetic_coarse_with_ml(
    encoder_path=None,
    decoder_path=None,
    percentiles_path=None,
    index=None,
    scaling_multiplier=500,
    pad_x_minus=6000,
    pad_x_plus=2000,
    pad_y=100,
    pixel_size_x=25,
    pixel_size_y=25,
    filename=None,
    uniform_thickness=2000,
    surface_slope=-50,
    u_in=50,
    u_out=None,
    constant_temperature=260,
    constant_C=1e-5,
    drichlet_ids=[1],
    side_wall_ids=[3, 4],
    nx=160,
    ny=80,
    plot_topography=False,
    plot_mesh=False,
    plot_depth_average_vel=False,
    z_samples=None,
    # --- ML ---
    model_path="xgb_shear_model",
    input_features=None,
    rms_roughness = 43.865605299393444, 
    slope_roughness = 0.03027835532924451,
    n_updates=10,
    tol_tau=1e-2,
    use_indicator_mean=True,
    debug_mode=True,
    use_ml_drag=True,
    # --- NEW knobs for Option B ---
    ramp_schedule=None,        # e.g., [0.05, 0.10, 0.20, 0.40, 0.80, 1.00]
    ramp_power=2.0,            # used only if ramp_schedule is None
    alpha0=0.05,               # used only if ramp_schedule is None; keeps k=0 mostly physics-only
    alpha_max=1.0,             # used only if ramp_schedule is None
    
):
    """
    Coarse-domain run with ML basal traction coupling.

    ML predicts (tau_x, tau_y) [Pa] given features; we add it to the Coulomb-like term:
        tau_total = -C * u/|u| + tau_ml_const

    Constant features (latent/roughness/slope-equiv) are set once; only (u,h) evolve
    inside the Picard coupling loop.
    """
    create_synthetic = CreateSynthetic()

    padded_topo, transform, latent_vector = create_synthetic.create_processed_topography(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        percentiles_path=percentiles_path,
        index=index,
        scaling_multiplier=scaling_multiplier,
        pad_x_minus=pad_x_minus,
        pad_x_plus=pad_x_plus,
        pad_y=pad_y,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        z_samples=z_samples,
    )

    if plot_topography:
        create_synthetic.plot_padded_topography(padded_topo)

    create_synthetic.setup_model(
        filename=filename,
        uniform_thickness=uniform_thickness,
        surface_slope=surface_slope,
        u_in=u_in,
        u_out=u_out,
        constant_temperature=constant_temperature,
        constant_C=constant_C,
        drichlet_ids=drichlet_ids,
        side_wall_ids=side_wall_ids,
        nx=nx,
        ny=ny,
    )

    if plot_mesh:
        fig, axes = icepack.plot.subplots()
        firedrake.triplot(create_synthetic.mesh2d, axes=axes)
        axes.set_title("Coarse mesh")

    # --- Load ML traction model ---
    if input_features is None:
        raise ValueError("input_features must be provided in the exact training order.")
    create_synthetic.load_ml_drag_model(model_path=model_path, input_features=input_features)

    # --- Constant features for this coarse run ---
    # rms_roughness, slope_roughness = create_synthetic.compute_roughness_metrics(create_synthetic.b)

    # surface_slope_equiv computed from current box drop + training calibration factor
    # delta_s_box_current = create_synthetic.box_surface_drop_x(create_synthetic.s0, strip_frac=strip_frac)
    # surface_slope_equiv = float(delta_s_box_current / float(k_train_slope))

    #print("RMS roughness (m):", rms_roughness, "Slope roughness (m/m):", slope_roughness)

    # latent_vector returned by create_processed_topography is shape (1,latent_dim) typically
    lv = np.asarray(latent_vector).reshape(-1)

    create_synthetic.set_ml_constant_features(
        latent_vector=lv,
        rms_roughness=float(rms_roughness),
        slope_roughness=float(slope_roughness),
    )

    # --- Run coupled coarse diagnostic solve with outer ML updates ---
    u0 = create_synthetic.u0
    if debug_mode:
        create_synthetic.debug_tau = debug_mode
    u_sol = create_synthetic.coarse_diagnostic_solve_with_ml_updates(
        u_init=u0,
        h=create_synthetic.h0,
        s=create_synthetic.s0,
        A=create_synthetic.A,
        C=create_synthetic.C,
        b=create_synthetic.b,
        temperature_K=float(constant_temperature),
        C_value=float(constant_C),
        n_updates=int(n_updates),
        use_indicator_mean=use_indicator_mean,
        tol_tau=float(tol_tau),
        verbose=True,
        update_use_ml_drag = use_ml_drag,
        # --- NEW knobs for Option B ---
        ramp_schedule=ramp_schedule,        # e.g., [0.05, 0.10, 0.20, 0.40, 0.80, 1.00]
        ramp_power=ramp_power,            # used only if ramp_schedule is None
        alpha0=alpha0,               # used only if ramp_schedule is None; keeps k=0 mostly physics-only
        alpha_max=alpha_max,             # used only if ramp_schedule is None
    
    )

    if plot_depth_average_vel:
        fig, axes = icepack.plot.subplots()
        colors = firedrake.tripcolor(icepack.depth_average(u_sol), axes=axes)
        fig.colorbar(colors, ax=axes, fraction=0.012, pad=0.04)

    return u_sol, create_synthetic, latent_vector


def create_data(nx = 48, ny = 32):
    thickness = random.randint(2000, 5000)
    surface_slope = random.randint(-100, -50)
    u_in = random.randint(5, 500)
    temperature = random.randint(243, 265)
    #c = random.uniform(0.01, 0.00001)
    c = 10**random.uniform(-5, -2) # want log-uniform, otherwise you massively overweight high C values which have much higher shear stress and dominate the training
    u0, create_synthetic, latent_vector = run_synthetic_high_res(encoder_path='vae/encoder_model', 
                                              decoder_path='vae/decoder_model', 
                                              percentiles_path='vae/latent_percentiles.pkl',
                           index = None, 
                           scaling_multiplier = 500, 
                           pad_x_minus= 600, 
                           pad_x_plus = 200, 
                           pad_y = 10, 
                           pixel_size_x = 25, 
                           pixel_size_y = 25, 
                           filename = None, 
                           uniform_thickness=thickness, 
                           surface_slope=surface_slope, 
                           u_in = u_in, 
                           u_out = None, 
                           constant_temperature = temperature, 
                           constant_C=c, 
                           drichlet_ids = [1], 
                           side_wall_ids = [3, 4],
                           nx = nx, ny = ny,
                           plot_topography = False,
                           plot_mesh = False,
                           plot_depth_average_vel= False)
    #total_form_drag_ds, shear_stress_ds = create_synthetic.compute_form_drag_volume(u0, create_synthetic.h0, create_synthetic.s0, create_synthetic.b)
    #bed_alongflow_stress, bed_vec_stress = create_synthetic.compute_form_drag_bp(u0, create_synthetic.s0, create_synthetic.b)
    bed_alongflow_stress, bed_vec_stress = create_synthetic.compute_basal_total_drag_hybrid(u = u0, s= create_synthetic.s0, A = create_synthetic.A, b = create_synthetic.b, h = create_synthetic.h0)

    rms_roughness, slope_roughness = create_synthetic.compute_roughness_metrics(create_synthetic.b)    
    mean_velocity_area,  mean_velocity_thickness = create_synthetic.compute_mean_velocities(u0, create_synthetic.h0)
    mean_thickness = create_synthetic.compute_mean_thickness(create_synthetic.h0)
    mean_surface = create_synthetic.compute_mean_surface_height(create_synthetic.s0)
    mean_s_slope_x_y, mean_s_slope = create_synthetic.compute_surface_slope_metrics(create_synthetic.s0)
    # print("Latent Vector:", latent_vector[0])
    list_of_parameters = [mean_thickness, mean_surface, mean_s_slope_x_y, mean_s_slope, u_in, temperature, c, bed_alongflow_stress, bed_vec_stress[0], bed_vec_stress[1], rms_roughness, slope_roughness, mean_velocity_area[0], mean_velocity_area[1], mean_velocity_thickness[0], mean_velocity_thickness[1]] + list(latent_vector[0])
    # print("Parameters:", list_of_parameters)
    return list_of_parameters, create_synthetic, u0, latent_vector

def create_dataset(n_samples = 10, nx = 48, ny = 32, dataset_name='synthetic_ice_flow_parameters.csv'):
    all_parameters = []
    for i in range(n_samples):
        print(f"Creating sample {i+1} of {n_samples}")
        try:
            list_of_parameters, create_synthetic, u0, latent_vector = create_data(nx=nx, ny=ny)
            all_parameters.append(list_of_parameters)
            # Save the velocity field and bed topography for each sample
            firedrake.File(f"output_data/velocity_field_{i}.pvd").write(u0)
            firedrake.File(f"output_data/bed_topography_{i}.pvd").write(create_synthetic.b)
        except Exception as e:
            print(f"Sample {i+1} failed with error: {e}. Skipping.")
    columns = ['Thickness (m)', 'Surface Height (m)', 'Surface Slope X Y', 'Surface Slope', 'Inflow Velocity (m/a)', 'Temperature (K)', 'C', 'Bed Along-Flow Stress (Pa)', 'Bed Vector Stress X (Pa)', 'Bed Vector Stress Y (Pa)', 'RMS Roughness (m)', 'Slope Roughness', 'x-Mean Velocity (m/a)', 'y-Mean Velocity (m/a)', 'x-Mean Velocity h (m/a)', 'y-Mean Velocity h (m/a)']+ [f'latent_{j}' for j in range(len(latent_vector[0]))]
    df = pd.DataFrame(all_parameters, columns=columns)
    df.to_csv(f'output_data/{dataset_name}', index=False)
    print(f"Dataset creation complete. Parameters saved to 'output_data/{dataset_name}'.")
    return df

def worker(args):
    i, n_samples = args
    print(f"Creating sample {i+1} of {n_samples}")
    list_of_parameters, create_synthetic, u0 = create_data()
    # Save the velocity field and bed topography for each sample
    firedrake.File(f"output_data/velocity_field_{i}.pvd").write(u0)
    firedrake.File(f"output_data/bed_topography_{i}.pvd").write(create_synthetic.b)
    # Explicitly delete objects and collect garbage
    del create_synthetic, u0
    gc.collect()
    return list_of_parameters

def create_dataset_parallel(n_samples=10, dataset_name='synthetic_ice_flow_parameters.csv'):
    all_parameters = []

    args_list = [(i, n_samples) for i in range(n_samples)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(worker, args_list))

    columns = ['Thickness (m)', 'Surface Slope (m)', 'Inflow Velocity (m/a)', 'Temperature (K)', 'C', 'Bed Along-Flow Stress (Pa)', 'Bed Vector Stress X (Pa)', 'Bed Vector Stress Y (Pa)', 'RMS Roughness (m)', 'Slope Roughness', 'x-Mean Velocity (m/a)', 'y-Mean Velocity (m/a)']+ [f'latent_{j}' for j in range(len(latent_vector[0]))]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f'output_data/{dataset_name}', index=False)
    print(f"Dataset creation complete. Parameters saved to 'output_data/{dataset_name}'.")
    return df

def process_df(cutoff_velocity=1200, datafile='synthetic_ice_flow_parameters_test.csv'):
    df = pd.read_csv(f'output_data/{datafile}')
    # Split 'Mean Velocity (m/a)' column into 'Vel-x' and 'Vel-y' for all rows
    # if 'Mean Velocity (m/a)' in df.columns:
    #     if isinstance(df['Mean Velocity (m/a)'].iloc[0], (list, tuple, np.ndarray, str)):
    #         # If stored as string, convert to list
    #         if isinstance(df['Mean Velocity (m/a)'].iloc[0], str):
    #             df['Mean Velocity (m/a)'] = df['Mean Velocity (m/a)'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    #         df[['Vel-x', 'Vel-y']] = pd.DataFrame(df['Mean Velocity (m/a)'].tolist(), index=df.index)
    # feature_cols = [
    #     'Thickness (m)', 'Surface Slope (m)', 'Inflow Velocity (m/a)', 'Temperature (K)',
    #     'C (Pa m^(1/3) a^(1/3))', 'Total Form Drag (Pa)', 'RMS Roughness (m)',
    #     'Slope Roughness', 'Mean Velocity (m/a)'
    # ]

    filtered_df = df[((df['x-Mean Velocity (m/a)'].abs() < cutoff_velocity) & 
                  (df['y-Mean Velocity (m/a)'].abs() < cutoff_velocity))]

    num_not_selected = len(df) - len(filtered_df)
    print(f"Number of rows that didn't make the cut: {num_not_selected}")
    return filtered_df

def plot_corelation_matrix(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Feature Correlation Matrix')
    plt.show()

def scatter_plot_features_vs_target(df, feature_cols, target_col='Shear Stress (Pa)'):
    num_features = len(feature_cols)
    cols = 3
    rows = (num_features + cols - 1) // cols  # Calculate number of rows needed

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, feature in enumerate(feature_cols):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(df[feature], df[target_col], alpha=0.7)
        plt.xlabel(feature)
        plt.ylabel(target_col)
        plt.title(f'{feature} vs {target_col}')
    plt.tight_layout()
    plt.show()

def predict_shear_stress_xgb(input_features, n_estimators=200, display_stats=False, cutoff_velocity=1200, target_col=['Bed Vector Stress X (Pa)', 'Bed Vector Stress Y (Pa)'], datafile='synthetic_ice_flow_parameters_test.csv', save_model=False, model_path='xgb_shear_model'):
    """
    Reads the synthetic ice flow parameters CSV, trains an XGBoost regressor to predict shear stress,
    and returns the predicted value for the given input features.
    input_features: list or array of 9 values matching the feature columns order.
    """
    filtered_df = process_df(cutoff_velocity=cutoff_velocity, datafile=datafile)

    
    X = filtered_df[input_features]
    if isinstance(target_col, (list, tuple)):
        y = filtered_df[target_col].values
    else:
        y = filtered_df[target_col].values.reshape(-1, 1)
    # summary statistics
    if display_stats:
        print("Feature statistics:")
        print(X.describe())
    scaler_input = RobustScaler()
    scaler_output = RobustScaler()
    X = scaler_input.fit_transform(X)
    if y.ndim > 1 and y.shape[1] > 1:
        y = scaler_output.fit_transform(y)
    else:
        y = scaler_output.fit_transform(y).flatten()
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = xgb.XGBRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Validation and test predictions
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    if save_model:
        # Save XGBoost model using native format (recommended)
        model.save_model(f'{model_path}.json')
        
        # Save scalers using joblib
        joblib.dump(scaler_input, f'{model_path}_scaler_input.pkl')
        joblib.dump(scaler_output, f'{model_path}_scaler_output.pkl')
        
        print(f"Model saved to {model_path}.json")

    # If output is vector, plot each dimension separately
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        n_outputs = y_test.shape[1]
        fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 6))
        for i in range(n_outputs):
            r2_test = r2_score(y_test[:, i], y_test_pred[:, i])
            ax = axes[i] if n_outputs > 1 else axes
            ax.scatter(y_test[:, i], y_test_pred[:, i], alpha=0.7, label='Test')
            ax.scatter(y_val[:, i], y_val_pred[:, i], alpha=0.7, label='Validation')
            ax.plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()], 'r--', label='Ideal')
            ax.set_xlabel('True Shear Stress (Pa)')
            ax.set_ylabel('Predicted Shear Stress (Pa)')
            ax.set_title(f'XGBoost Shear Stress Prediction\nDim {i+1} R2 = {r2_test:.3f}')
            ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(6,6))
        r2_test = r2_score(y_test, y_test_pred)
        plt.scatter(y_test, y_test_pred, alpha=0.7, label='Test')
        plt.scatter(y_val, y_val_pred, alpha=0.7, label='Validation')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
        plt.xlabel('True Shear Stress (Pa)')
        plt.ylabel('Predicted Shear Stress (Pa)')
        plt.title(f'XGBoost Shear Stress Prediction \nR2 = {r2_test:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

def load_shear_stress_model(model_path='xgb_shear_model'):
    """Load saved XGBoost model and scalers"""
    model = xgb.XGBRegressor()
    model.load_model(f'{model_path}.json')
    
    scaler_input = joblib.load(f'{model_path}_scaler_input.pkl')
    scaler_output = joblib.load(f'{model_path}_scaler_output.pkl')
    
    return model, scaler_input, scaler_output

def plot_loaded_xgb_shear_stress(
    input_features,
    cutoff_velocity=1200,
    target_col=['Bed Vector Stress X (Pa)', 'Bed Vector Stress Y (Pa)'],
    datafile='synthetic_ice_flow_parameters_test.csv',
    model_path='xgb_shear_model'
):
    """
    Load a saved XGBoost shear-stress model and scalers, re-run predictions
    on the same processed dataset, and plot true vs predicted scatter.

    This is intended to verify that the saved model was loaded correctly.
    """

    # --- Load model and scalers ---
    model = xgb.XGBRegressor()
    model.load_model(f'{model_path}.json')

    scaler_input = joblib.load(f'{model_path}_scaler_input.pkl')
    scaler_output = joblib.load(f'{model_path}_scaler_output.pkl')

    # --- Load and process data exactly as during training ---
    filtered_df = process_df(
        cutoff_velocity=cutoff_velocity,
        datafile=datafile
    )

    X = filtered_df[input_features]

    if isinstance(target_col, (list, tuple)):
        y = filtered_df[target_col].values
    else:
        y = filtered_df[target_col].values.reshape(-1, 1)

    # --- Apply saved scalers ---
    X_scaled = scaler_input.transform(X)

    if y.ndim > 1 and y.shape[1] > 1:
        y_scaled = scaler_output.transform(y)
    else:
        y_scaled = scaler_output.transform(y).flatten()

    # --- Reproduce the same split ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # --- Predictions ---
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # --- Plot ---
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        n_outputs = y_test.shape[1]
        fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 6))

        for i in range(n_outputs):
            r2_test = r2_score(y_test[:, i], y_test_pred[:, i])
            ax = axes[i] if n_outputs > 1 else axes

            ax.scatter(y_test[:, i], y_test_pred[:, i], alpha=0.7, label='Test')
            ax.scatter(y_val[:, i], y_val_pred[:, i], alpha=0.7, label='Validation')
            ax.plot(
                [y_test[:, i].min(), y_test[:, i].max()],
                [y_test[:, i].min(), y_test[:, i].max()],
                'r--',
                label='Ideal'
            )

            ax.set_xlabel('True Shear Stress (Pa)')
            ax.set_ylabel('Predicted Shear Stress (Pa)')
            ax.set_title(f'Loaded XGBoost Model\nDim {i+1} R² = {r2_test:.3f}')
            ax.legend()

        plt.tight_layout()
        plt.show()

    else:
        r2_test = r2_score(y_test, y_test_pred)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.7, label='Test')
        plt.scatter(y_val, y_val_pred, alpha=0.7, label='Validation')
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--',
            label='Ideal'
        )

        plt.xlabel('True Shear Stress (Pa)')
        plt.ylabel('Predicted Shear Stress (Pa)')
        plt.title(f'Loaded XGBoost Shear Stress Prediction\nR² = {r2_test:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.show()



def predict_shear_stress_mlp(
    input_features,
    layers_config=[64, 64],
    activation='swish',
    epochs=100,
    batch_size=32,
    patience=10,
    min_lr=1e-5,
    display_stats=False,
    cutoff_velocity=1200,
    target_col = 'Shear Stress (Pa)',
    datafile='synthetic_ice_flow_parameters_test.csv'):
    """
    Train an MLP regressor using Keras/TensorFlow to predict shear stress.
    layers_config: list of neuron counts per layer
    activation: activation function for hidden layers
    epochs: max epochs
    batch_size: batch size
    patience: early stopping patience
    min_lr: minimum learning rate for ReduceLROnPlateau
    """
    filtered_df = process_df(cutoff_velocity=cutoff_velocity, datafile=datafile)
    X = filtered_df[input_features].values
    if isinstance(target_col, (list, tuple)):
        y = filtered_df[target_col].values
    else:
        y = filtered_df[target_col].values.reshape(-1, 1)
    # summary statistics
    if display_stats:
        print("Feature statistics:")
        print(pd.DataFrame(X, columns=input_features).describe())
    scaler_input = RobustScaler()
    scaler_output = RobustScaler()
    X = scaler_input.fit_transform(X)
    y = scaler_output.fit_transform(y)
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build model
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    for units in layers_config:
        model.add(layers.Dense(units, activation=activation))
    if y.ndim > 1 and y.shape[1] > 1:
        model.add(layers.Dense(y.shape[1], activation='linear'))
    else:
        model.add(layers.Dense(1, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mae'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(2, patience//2), min_lr=min_lr)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Validation and test predictions
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # If output is vector, plot each dimension separately
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        n_outputs = y_test.shape[1]
        fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 6))
        for i in range(n_outputs):
            r2_test = r2_score(y_test[:, i], y_test_pred[:, i])
            ax = axes[i] if n_outputs > 1 else axes
            ax.scatter(y_test[:, i], y_test_pred[:, i], alpha=0.7, label='Test')
            ax.scatter(y_val[:, i], y_val_pred[:, i], alpha=0.7, label='Validation')
            ax.plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()], 'r--', label='Ideal')
            ax.set_xlabel('True Shear Stress (Pa)')
            ax.set_ylabel('Predicted Shear Stress (Pa)')
            ax.set_title(f'MLP Shear Stress Prediction\nDim {i+1} R2 = {r2_test:.3f}')
            ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        y_val_pred = y_val_pred.flatten()
        y_test_pred = y_test_pred.flatten()
        plt.figure(figsize=(6,6))
        r2_test = r2_score(y_test, y_test_pred)
        plt.scatter(y_test, y_test_pred, alpha=0.7, label='Test')
        plt.scatter(y_val, y_val_pred, alpha=0.7, label='Validation')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
        plt.xlabel('True Shear Stress (Pa)')
        plt.ylabel('Predicted Shear Stress (Pa)')
        plt.title(f'MLP Shear Stress Prediction\nR2 = {r2_test:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model, scaler_input, scaler_output