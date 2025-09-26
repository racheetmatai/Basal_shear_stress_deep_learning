import matplotlib.pyplot as plt
import firedrake
import icepack
from src.create_synthetic import CreateSynthetic


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

def run_synthetic_high_res(model_filename = None, 
                           index = None, 
                           scaling_multiplier = 50, 
                           pad_x_minus= 100, 
                           pad_x_plus = 100, 
                           pad_y = 10, 
                           pixel_size_x = 50, 
                           pixel_size_y = 50, 
                           filename = None, 
                           uniform_thickness=5000, 
                           surface_slope=-300, 
                           u_in = 200, 
                           u_out = None, 
                           constant_temperature = 260, 
                           constant_C=0.01, 
                           drichlet_ids = [1], 
                           side_wall_ids = [3, 4],
                           plot_topography = False,
                           plot_mesh = False,
                           plot_depth_average_vel = False):
    """
    Runs a high-resolution synthetic ice flow model.
    """
    create_synthetic = CreateSynthetic()
    padded_topo, transform = create_synthetic.create_processed_topography(model_filename = model_filename, index = index, scaling_multiplier = scaling_multiplier, pad_x_minus= pad_x_minus, pad_x_plus = pad_x_plus, pad_y = pad_y, pixel_size_x = pixel_size_x, pixel_size_y = pixel_size_y)
    if plot_topography:
        create_synthetic.plot_padded_topography(padded_topo)
    create_synthetic.setup_model(filename = filename, uniform_thickness=uniform_thickness, surface_slope=surface_slope, u_in = u_in, u_out = u_out, constant_temperature = constant_temperature, constant_C=constant_C, drichlet_ids = drichlet_ids, side_wall_ids = side_wall_ids)
    if plot_mesh:
        plot_high_res_mesh(create_synthetic)
    u = create_synthetic.diagnostic_solve(create_synthetic.u0, create_synthetic.h0, create_synthetic.s0, create_synthetic.A, create_synthetic.C, create_synthetic.b)
    if plot_depth_average_vel:
        plot_depth_average_u(u)
    return u, create_synthetic

