import subprocess
import geojson
import icepack
import firedrake

def fetch_outline(name):
    """
    Fetches outline data for a given name (GeoJSON with LineStrings or Polygons).

    :param name: Path to the GeoJSON file.
    :return: Parsed GeoJSON outline data (dict).
    """
    try:
        print("Reading local outline")
        outline_filename = name

        # Always open as UTF-8, ignore/replace any bad characters
        with open(outline_filename, "r", encoding="utf-8", errors="replace") as outline_file:
            outline = geojson.load(outline_file)

        return outline

    except Exception as e:
        print(f"Error fetching outline: {e}")
        return None

def create_geo_file(outline, name='pig', lcar = 10e3):
    """
    Creates a geo file from the given outline.

    :param outline: GeoJSON outline data.
    :param name: Name for the geo file (default is 'pig').
    """
    if "." in name:
        name = name.split('.')[0]
    geo_name = name + '.geo'
    try:
        geometry = icepack.meshing.collection_to_geo(outline,  lcar=lcar)
        with open(geo_name, "w") as geo_file:
            geo_file.write(geometry.get_code())
    except Exception as e:
        print(f"Error creating geo file: {e}")

def create_mesh(outline, name='pig', lcar = 10e3, **kwargs):
    """
    Creates a mesh using Gmsh.

    :param outline: GeoJSON outline data.
    :param name: Name for the mesh (default is 'pig').
    :param kwargs: Additional keyword arguments for Gmsh.
    """
    create_geo_file(outline, name, lcar)
    mesh_name = name + '.msh'
    geo_name = name + '.geo'
    try:
        command = f"gmsh -2 -format msh2 -v 2 -o {mesh_name} {geo_name}"
        subprocess.run(command.split(), **kwargs)
    except Exception as e:
        print(f"Error creating mesh: {e}")

def check_available_outlines():
    """
    Checks and returns the available glacier names in the icepack datasets.

    :return: List of available glacier names.
    """
    return icepack.datasets.get_glacier_names()

def create_rectangle_mesh(nx, ny, Lx, Ly, originX=2000.0, originY=2000.0):
    """
    Creates a rectangular mesh.

    :param nx: Number of elements in the x-direction.
    :param ny: Number of elements in the y-direction.
    :param Lx: Length of the domain in the x-direction.
    :param Ly: Length of the domain in the y-direction.
    :param originX: X-coordinate of the origin (default is 2000.0).
    :param originY: Y-coordinate of the origin (default is 2000.0).
    :return: Firedrake mesh object.
    """
    mesh2d = firedrake.RectangleMesh(
        nx, ny, Lx, Ly, originX=originX, originY=originY
    )
    return mesh2d

def get_bbox_from_outline(outline):
    coords = []
    for feature in outline["features"]:
        geom = feature["geometry"]
        if geom["type"] == "LineString":
            coords.extend(geom["coordinates"])
    
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    originX = min(xs)
    originY = min(ys)
    Lx = max(xs) - originX
    Ly = max(ys) - originY
    return Lx, Ly, originX, originY


# Example usage:
if __name__ == "__main__":
    outline_data = fetch_outline()
    if outline_data:
        create_mesh(outline_data)