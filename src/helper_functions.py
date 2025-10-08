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
                           plot_depth_average_vel = False):
    """
    Runs a high-resolution synthetic ice flow model.
    """
    create_synthetic = CreateSynthetic()
    padded_topo, transform, latent_vector = create_synthetic.create_processed_topography(encoder_path=encoder_path, decoder_path=decoder_path, percentiles_path=percentiles_path, index=index, scaling_multiplier=scaling_multiplier, pad_x_minus=pad_x_minus, pad_x_plus=pad_x_plus, pad_y=pad_y, pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y)
    if plot_topography:
        create_synthetic.plot_padded_topography(padded_topo)
    create_synthetic.setup_model(filename = filename, uniform_thickness=uniform_thickness, surface_slope=surface_slope, u_in = u_in, u_out = u_out, constant_temperature = constant_temperature, constant_C=constant_C, drichlet_ids = drichlet_ids, side_wall_ids = side_wall_ids, nx = nx, ny = ny)
    if plot_mesh:
        plot_high_res_mesh(create_synthetic)
    u = create_synthetic.diagnostic_solve(create_synthetic.u0, create_synthetic.h0, create_synthetic.s0, create_synthetic.A, create_synthetic.C, create_synthetic.b)
    if plot_depth_average_vel:
        plot_depth_average_u(u)
    return u, create_synthetic, latent_vector

def create_data(nx = 48, ny = 32):
    thickness = random.randint(1000, 3000)
    surface_slope = random.randint(-50, -20)
    u_in = random.randint(20, 50)
    temperature = random.randint(243, 265)
    c = random.uniform(0.01, 0.0001)
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
    total_form_drag, shear_stress = create_synthetic.compute_form_drag_volume(u0, create_synthetic.h0, create_synthetic.s0, create_synthetic.b)
    rms_roughness, slope_roughness = create_synthetic.compute_roughness_metrics(create_synthetic.b)    
    mean_velocity = create_synthetic.compute_mean_velocity(u0, create_synthetic.h0)
    # print("Latent Vector:", latent_vector[0])
    list_of_parameters = [thickness, surface_slope, u_in, temperature, c, total_form_drag, shear_stress, rms_roughness, slope_roughness, mean_velocity[0], mean_velocity[1]] + list(latent_vector[0])
    # print("Parameters:", list_of_parameters)
    return list_of_parameters, create_synthetic, u0, latent_vector

def create_dataset(n_samples = 10, dataset_name='synthetic_ice_flow_parameters.csv'):
    all_parameters = []
    for i in range(n_samples):
        print(f"Creating sample {i+1} of {n_samples}")
        try:
            list_of_parameters, create_synthetic, u0, latent_vector = create_data()
            all_parameters.append(list_of_parameters)
            # Save the velocity field and bed topography for each sample
            firedrake.File(f"output_data/velocity_field_{i}.pvd").write(u0)
            firedrake.File(f"output_data/bed_topography_{i}.pvd").write(create_synthetic.b)
        except Exception as e:
            print(f"Sample {i+1} failed with error: {e}. Skipping.")
    columns = ['Thickness (m)', 'Surface Slope (m)', 'Inflow Velocity (m/a)', 'Temperature (K)', 'C (Pa m^(1/3) a^(1/3))', 'Total Form Drag (Pa)', 'Shear Stress (Pa)', 'RMS Roughness (m)', 'Slope Roughness', 'x-Mean Velocity (m/a)', 'y-Mean Velocity (m/a)']+ [f'latent_{j}' for j in range(len(latent_vector[0]))]
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

    columns = ['Thickness (m)', 'Surface Slope (m)', 'Inflow Velocity (m/a)', 'Temperature (K)', 'C (Pa m^(1/3) a^(1/3))', 'Total Form Drag (Pa)', 'Shear Stress (Pa)', 'RMS Roughness (m)', 'Slope Roughness', 'Mean Velocity (m/a)']
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

def predict_shear_stress_xgb(input_features, n_estimators=200, display_stats=False):
    """
    Reads the synthetic ice flow parameters CSV, trains an XGBoost regressor to predict shear stress,
    and returns the predicted value for the given input features.
    input_features: list or array of 9 values matching the feature columns order.
    """
    filtered_df = process_df()

    target_col = 'Shear Stress (Pa)'
    X = filtered_df[input_features]
    # summary statistics
    if display_stats:
        print("Feature statistics:")
        print(X.describe())
    scaler_input = RobustScaler()
    scaler_output = RobustScaler()
    X = scaler_input.fit_transform(filtered_df[input_features])
    y = scaler_output.fit_transform(filtered_df[target_col].values.reshape(-1, 1)).flatten()
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = xgb.XGBRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Validation and test predictions
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Plot x-y performance for test set
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
