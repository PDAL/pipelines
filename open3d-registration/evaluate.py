import io
import json
import numpy as np
from numpy.lib import recfunctions as rfn
import open3d as o3d
import pdal
from math import sin, cos, radians
from scipy.spatial.transform import Rotation

moving_filename = "p2at_met_048.laz"
fixed_filename = "box_map.laz"
aligned_filename = "p048_aligned.laz"
xformed_filename = "p048_xformed.laz"

def random_vector(min_magnitude, max_magnitude):
    [inclination, azimuth] = np.random.uniform(0, 360, 2)
    radius = np.random.uniform(min_magnitude, max_magnitude)
    x = radius * sin(radians(inclination)) * cos(radians(azimuth))
    y = radius * sin(radians(inclination)) * sin(radians(azimuth))
    z = radius * cos(radians(inclination))
    vec = [x, y, z]
    return vec


def random_transform(lower_t, upper_t, lower_angle, upper_angle):
    trans = random_vector(lower_t, upper_t)
    rot = random_vector(radians(lower_angle), radians(upper_angle))
    rot = Rotation.from_rotvec(rot)
    return trans, rot

def array2string(array):
    bio = io.BytesIO()
    np.savetxt(bio, array, fmt='%f', newline=' ')
    mystr = bio.getvalue().decode('latin1')
    return mystr

# Read the fixed point cloud so that we can later get it's center coords
f = pdal.Reader(fixed_filename).pipeline()
f.execute()
coords = rfn.structured_to_unstructured(f.arrays[0][['X', 'Y', 'Z']])
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(coords)

# Create the reader and pipeline in one step
p = pdal.Reader(moving_filename).pipeline()

# For evaluation purposes, center the moving cloud by the fixed cloud center coords
T0 = np.eye(4)
T0[0:3, 3] = -cloud.get_center()
T0str = array2string(T0)
#p |= pdal.Filter.transformation(matrix=T0str)

# Artificially introduce a random transformation for evaluation purposes
trans, rot = random_transform(-100,100,-45,45)
T = np.eye(4)
T[0:3,0:3] = rot.as_matrix()
T[0:3,3] = trans
Tstr = array2string(T)
print(Tstr)
Tstr = "0.9853426649066137 0.07183643076032609 0.15472349508187008 4.324860030404999 -0.04341682378658787 0.982747722195347 -0.17978290778642264 -2.5021918997317036 -0.16496912476890027 0.17043016674131917 0.9714621691746462 6.611935362864005 0 0 0 1"
p |= pdal.Filter.transformation(matrix=Tstr)

# Now, "uncenter" the cloud again
T1 = np.eye(4)
T1[0:3, 3] = cloud.get_center()
T1str = array2string(T1)
#p |= pdal.Filter.transformation(matrix=T1str)

# Setup the registration step as a Python filter
p |= pdal.Filter.python(
    script="/registration.py", function="filter", module="all",
    pdalargs=json.dumps({
        "threshold": 10.0,
        "global_alignment": False,
        "global_alignment_voxel_size": 3.0,
        "method": "feat",
        "max_iters": 30,
        "voxel_size": 0.1,
        "filename": fixed_filename
    }))

# Write the aligned point cloud
p |= pdal.Writer(aligned_filename, forward="all")

p.execute()

print(p.log)
print(json.loads(p.metadata)['metadata']['filters.python']['children'])

# For visualization, write the transformed cloud
q = pdal.Reader(moving_filename).pipeline()
q |= pdal.Filter.transformation(matrix=Tstr)
q |= pdal.Writer(xformed_filename, forward="all")
q.execute()

#f0_coords = rfn.structured_to_unstructured(f.arrays[0][['X', 'Y', 'Z']])
#p0_coords = rfn.structured_to_unstructured(p.arrays[0][['X', 'Y', 'Z']])
#q0_coords = rfn.structured_to_unstructured(q.arrays[0][['X', 'Y', 'Z']])

#weights = np.linalg.norm(f0_coords - cloud.get_center(), 2, axis=1)

#distances = np.linalg.norm(f0_coords - p0_coords, 2, axis=1)/len(weights)
#print(np.sum(distances/weights))

#distances = np.linalg.norm(f0_coords - q0_coords, 2, axis=1)/len(weights)
#print(np.sum(distances/weights))

src_original = pdal.Reader(moving_filename).pipeline()
src_original.execute()
f0_coords = rfn.structured_to_unstructured(src_original.arrays[0][['X', 'Y', 'Z']])
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(f0_coords)
p0_coords = rfn.structured_to_unstructured(p.arrays[0][['X', 'Y', 'Z']])
weights = np.linalg.norm(f0_coords - cloud.get_center(), 2, axis=1)
distances = np.linalg.norm(f0_coords - p0_coords, 2, axis=1)/len(weights)
print("Fontana score between original \"truth\" and aligned result after perburbation: {}".format(np.sum(distances/weights)))
