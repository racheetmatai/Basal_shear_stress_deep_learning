import numpy as np
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
from firedrake import inner, as_vector, assemble, grad, div, conditional, le, ge, And, Mesh, inner
from icepack.calculus import FacetNormal
from icepack.models.hybrid import _pressure_approx
from operator import itemgetter
from matplotlib import pyplot as plt
#from firedrake import assemble, Constant, inner, grad, dx


class CreateSynthetic:
    def compute_mean_velocity(self, u, h):
        """
        Compute thickness-weighted mean velocity for the box region using self.indicator.
        :param u: Firedrake velocity Function (scalar or vector)
        :param h: Firedrake thickness Function
        :return: mean_velocity (scalar or vector)
        """
        if self.indicator is None:
            self.compute_box_indicator()

        denominator = assemble(h * self.indicator * firedrake.dx)
        # Check if u is vector-valued
        if hasattr(u, 'ufl_shape') and len(u.ufl_shape) == 1:
            # Vector velocity: compute each component
            mean_components = [assemble(h * u[i] * self.indicator * firedrake.dx) / denominator for i in range(u.ufl_shape[0])]
            mean_velocity = as_vector(mean_components)
        else:
            # Scalar velocity
            mean_velocity = assemble(h * u * self.indicator * firedrake.dx) / denominator
        return mean_velocity
    
    def compute_roughness_metrics(self, b):
        """
        Compute RMS roughness and slope roughness for the box region using self.indicator.
        :param b: Firedrake bed elevation Function
        :return: rms_roughness, slope_roughness
        """
        if self.indicator is None:
            self.compute_box_indicator()

        # Area of the box
        area = assemble(self.indicator * firedrake.dx)
        # Mean bed elevation
        mean_b = assemble(b * self.indicator * firedrake.dx) / area
        # RMS roughness (standard deviation)
        rms_roughness = (assemble((b - mean_b)**2 * self.indicator * firedrake.dx) / area)**0.5
        # Slope roughness (RMS of gradient magnitude)
        slope_roughness = (assemble(firedrake.inner(grad(b), grad(b)) * self.indicator * firedrake.dx) / area)**0.5
        return rms_roughness, slope_roughness

    def compute_form_drag_volume(self, u, h, s, b):
        """
        Compute net form drag into/out of the original box using a volume integral.
        This is robust for arbitrary mesh resolution and box location.
        :param u: Firedrake velocity Function
        :param h: Firedrake thickness Function
        :param s: Firedrake surface Function
        :param b: Firedrake bed Function
        :return: Net form drag (float)
        """
        if self.indicator is None:
            self.compute_box_indicator()

        stress = ρ_I * g * h * grad(s)
        # Net flux into/out of box (divergence of stress)
        net_form_drag = assemble(div(stress) * self.indicator * firedrake.dx)

        area = assemble(self.indicator * firedrake.dx)
        average_shear_stress = net_form_drag / area
        return net_form_drag, average_shear_stress
    
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
    
    def __init__(self):
        self.indicator = None

    def load_model(self, filename):
        # Implement the logic to load a model
        pass

    def create_topography(self, filename):
        """
        Create topography using a trained model.
        :param filename: Path to the trained model file.
        :param z_samples: Latent space samples.
        :return: Generated images.
        """
        decoder = self.load_model(filename+"_decoder.h5")
        with open(filename+"_z_samples.pkl", "rb") as f:
            z_samples = pickle.load(f)
        def generate_images(decoder, n=5):
            # Sample from the latent space
            #latent_dim = 128  # This is the latent_dim defined in the VAE model
            random_latent_vectors = z_samples[np.random.choice(len(z_samples), size=n)]

            # Decode to generate images
            generated_images = decoder.predict(random_latent_vectors)

            return generated_images
        new_images = generate_images(decoder, n=1)
        return new_images[0]
    
    def create_topography_scaled(self, filename = None, multiplier=1):
        #b = self.create_topography(filename = filename)
        b = np.load('thw_image_0.npy') # temp, use model
        return b[:,:,0] * multiplier
    
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

    def create_processed_topography(self, model_filename, index = None, scaling_multiplier = 500, pad_x_minus= 100, pad_x_plus = 100, pad_y = 10, pixel_size_x = 25, pixel_size_y = 25):
        """
        Create processed topography using a trained model.
        :param model_filename: Path to the trained model file.
        :param pad_x_minus: Padding size on the left side of the data.
        :param pad_x_plus: Padding size on the right side of the data.
        :param pad_y: Padding size on the top and bottom of the data.
        :param pixel_size_x: Pixel size in the x-direction.
        :param pixel_size_y: Pixel size in the y-direction.
        :return: Generated images.
        """
        b = self.create_topography_scaled(filename=model_filename, multiplier= scaling_multiplier)
        self.padded_topo = self.pad_topography(b, pad_x_minus, pad_x_plus, pad_y)
        if not index:
            self.random_index = np.random.randint(0, 100000)
        else:
            self.random_index = index
        self.filename=f"base_{self.random_index}.tif"
        self.geo_filename=f"boundary_{self.random_index}.geojson"
        transform = self.save_tif(self.padded_topo, pixel_size_x, pixel_size_y, filename=self.filename)
        self.create_geojson_bbox_from_tif(self.filename, self.geo_filename)
        self.padded_topo = self.pad_topography(b, pad_x_minus+1, pad_x_plus+1, pad_y+1)
        transform = self.save_tif(self.padded_topo, pixel_size_x, pixel_size_y, filename=self.filename)

        return self.padded_topo, transform

    def create_mesh_synthetic(self, filename = None, lcar = 1e3):
        """
        Create a mesh using the provided filename or a default one.
        :param filename: Path to the GeoJSON file.
        :param lcar: Mesh size parameter.
        """
        if not filename:
            filename = f"boundary_{self.random_index}.geojson"
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
            filename = f"boundary_{self.random_index}.geojson"

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

        b_expr = rasterio.open(f"base_{self.random_index}.tif")
        self.b = icepack.interpolate(b_expr, self.Q)

        if uniform_thickness is not None and surface_slope is None:
            print(f"Using uniform thickness: {uniform_thickness} m")
            s_expr = self.b + uniform_thickness

        elif uniform_thickness is not None and surface_slope is not None:
            print(f"Using starting thickness: {uniform_thickness} m")
            print(f"Using surface slope: {surface_slope}")
            s_expr = self.b + uniform_thickness + (x * surface_slope / self.Lx)

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

    def friction(self, **kwargs):
        """
        Friction model for the hybrid model.
        :param kwargs: Keyword arguments containing velocity, thickness, surface, and friction.
        :return: Friction term.
        """
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        s = kwargs["surface"]
        C = kwargs["friction"]

        p_W = ρ_W * g * firedrake.max_value(0, h - s)
        p_I = ρ_I * g * h
        ϕ = 1 - p_W / p_I
        return icepack.models.hybrid.bed_friction(
            velocity=u,
            friction=C * ϕ,
        )

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
        model = icepack.models.HybridModel(friction=self.friction, terminus = self.terminus)
        opts = {"dirichlet_ids": drichlet_ids, "side_wall_ids": side_wall_ids}
        self.solver = icepack.solvers.FlowSolver(model, **opts)

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

    def setup_model(self, filename = None, uniform_thickness = 500,surface_slope = -200, u_in = 20, u_out = 200, constant_temperature = 260, constant_C=1, drichlet_ids = [4], side_wall_ids = [1, 3]):
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
        self.create_rectangle_mesh_synthetic(filename = filename, nx = 48, ny = 32)
        self.create_function_space()
        self.set_initial_conditions(uniform_thickness=uniform_thickness, surface_slope=surface_slope, u_in=u_in, u_out=u_out, constant_temperature=constant_temperature, constant_C=constant_C)
        self.create_model(drichlet_ids=drichlet_ids, side_wall_ids=side_wall_ids)

    def create_synthetic_setup(self):
        # Implement the logic to create synthetic data
        pass

    def save_synthetic_data(self, filename):
        # Implement the logic to save synthetic data
        pass