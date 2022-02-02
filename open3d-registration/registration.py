import copy
import io
import json
import numpy as np
from numpy.lib import recfunctions as rfn
import open3d as o3d
import pdal


def prepareCloud(coords, labels):
    class_to_color_dict = {
        0: [0, 0, 1.0],
        1: [0, 1.0, 1.0],
        2: [0.0, 1.0, 0],
        3: [0, 0.0, 0.5],
        6: [1.0, 0, 0]
    }

    colors_out = list()
    for label in labels:
        if label in class_to_color_dict:
            colors_out.append(class_to_color_dict[label])
        else:
            print("Unknown classification label: {}".format(label))
            # use black for all other labels
            class_to_color_dict[label] = [0.0, 0.0, 0.0]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(coords)
    cloud.estimate_normals()
    cloud.colors = o3d.utility.Vector3dVector(np.array(colors_out))
    return cloud


def readMoving(ins):
    # Create Open3D PointCloud of the input data (moving) and estimate normals.
    coords = np.vstack([ins['X'], ins['Y'], ins['Z']]).transpose()
    if 'Classification' in ins:
        labels = ins['Classification']
    else:
        labels = np.zeros_like(ins['X'])
    moving = prepareCloud(coords, labels)
    return moving


def readFixed(filename):
    # Create Open3D PointCloud of the fixed data and estimate normals.
    p = pdal.Pipeline(json.dumps([filename]))
    p.execute()
    coords = rfn.structured_to_unstructured(p.arrays[0][['X', 'Y', 'Z']])
    if 'Classification' in p.arrays[0].dtype.names:
        labels = p.arrays[0]['Classification']
    else:
        labels = np.zeros_like(p.arrays[0]['X'])
    fixed = prepareCloud(coords, labels)
    return fixed


def array2string(array):
    bio = io.BytesIO()
    np.savetxt(bio, array, fmt='%f', newline=' ')
    mystr = bio.getvalue().decode('latin1')
    return mystr


def createMetadata(evaluation, result, center, method):
    transformation = array2string(result.transformation)
    centroid = array2string(center)

    out_metadata = {
        'name': 'registration_method', 'value': method,
        'type': 'string',
        'children':
        [{'name': 'initial_fitness', 'value': '{:.2f}'.format(
            evaluation.fitness),
          'type': 'double'},
         {'name': 'initial_rmse', 'value': '{:.3f}'.format(
             evaluation.inlier_rmse),
          'type': 'double'},
         {'name': 'aligned_fitness', 'value': '{:.2f}'.format(
             result.fitness),
          'type': 'double'},
         {'name': 'aligned_rmse', 'value': '{:.3f}'.format(
             result.inlier_rmse),
          'type': 'double'},
         {'name': 'aligned_transformation', 'value': transformation,
          'type': 'string'},
         {'name': 'centroid', 'value': centroid, 'type': 'string'}]}
    return out_metadata


def computeGlobalAlignment(moving, fixed, radius, threshold):
    moving_down = moving.voxel_down_sample(radius)
    radius_normal = radius * 2
    radius_feature = radius * 5
    moving_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))
    moving_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        moving_down, o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
    fixed_down = fixed.voxel_down_sample(radius)
    fixed_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))
    fixed_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        fixed_down, o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
    distance_threshold = radius * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        moving_down, fixed_down, moving_fpfh, fixed_fpfh, o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    fast_result = o3d.pipelines.registration.evaluate_registration(
        fixed, moving, threshold, result.transformation)
    return fast_result


def runFeat(
        moving, fixed, voxel_size, lambda_geometric, trans, max_iter,
        multi_scale=True):
    if multi_scale:
        radii = [voxel_size * 2, voxel_size, voxel_size*0.5]    # multi-scale
    else:
        radii = [voxel_size]

    temp_trans = trans
    for ir, radius in enumerate(radii):

        moving_copy = copy.deepcopy(moving)
        fixed_copy = copy.deepcopy(fixed)

        if radius and radius > 0:
            print(" - Downsample with a voxel size %.2f" % radius)
            source_down = moving_copy.voxel_down_sample(radius)
            target_down = fixed_copy.voxel_down_sample(radius)
        else:
            print(" - No voxel_down_sample")
            source_down = moving_copy
            target_down = fixed_copy

        print("num_moving_points = {}".format(len(source_down.points)))
        print("num_fixed_points = {}".format(len(target_down.points)))

        print(" - Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print(" - Applying feature-based point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source=source_down, target=target_down,
            max_correspondence_distance=radius, init=temp_trans,
            estimation_method=o3d.pipelines.registration.
            TransformationEstimationForColoredICP(
                lambda_geometric=lambda_geometric),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6,
                max_iteration=max_iter),)

        # intermediate results for multi-scale
        print(result_icp)
        temp_trans = result_icp.transformation

    return result_icp


def filter(ins, outs):
    global out_metadata

    # Setup some arguments. These could really be pdalargs.
    try:
        threshold = float(pdalargs['threshold'])
    except KeyError:
        threshold = 10.0

    try:
        max_iters = int(pdalargs['max_iters'])
    except KeyError:
        max_iters = 50

    try:
        voxel_size = float(pdalargs['voxel_size'])
    except KeyError:
        voxel_size = 1.0

    try:
        global_alignment_voxel_size = float(
            pdalargs['global_alignment_voxel_size'])
    except KeyError:
        voxel_size = 3.0

    try:
        filename = pdalargs['filename']
    except KeyError:
        print("Missing filename!")
        return False

    trans_init = np.eye(4)

    moving = readMoving(ins)
    fixed = readFixed(filename)

    # Center both clouds using the center of the fixed data.
    center = fixed.get_center()
    fixed.translate(-center)
    moving.translate(-center)

    # Evaluate initial registration.
    evaluation = o3d.pipelines.registration.evaluate_registration(
        fixed, moving, threshold, trans_init)

    if pdalargs['global_alignment']:
        fast_result = computeGlobalAlignment(
            moving, fixed, global_alignment_voxel_size, threshold)
        print(fast_result.fitness)
        print(fast_result.transformation)
        trans_init = fast_result.transformation

    if pdalargs['method'] == 'point2plane':
        # Apply point to plane ICP.
        result = o3d.pipelines.registration.registration_icp(
            moving, fixed, threshold, trans_init, o3d.pipelines.registration.
            TransformationEstimationPointToPlane())

    elif pdalargs['method'] == 'point2point':
        result = o3d.pipelines.registration.registration_icp(
            moving, fixed, threshold, trans_init, o3d.pipelines.registration.
            TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iters))

    elif pdalargs['method'] == 'feat':
        print("feat")
        print(trans_init)
        result = runFeat(
            moving=moving, fixed=fixed, voxel_size=voxel_size,
            lambda_geometric=0.968, trans=trans_init, max_iter=max_iters,
            multi_scale=True)

    if pdalargs['method'] == 'generalized':
        # Apply Generalized-ICP.
        result = o3d.pipelines.registration.registration_generalized_icp(
            moving, fixed, threshold, trans_init, o3d.pipelines.registration.
            TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iters))

    # Extract and save transformation.
    out_metadata = createMetadata(
        evaluation, result, center, pdalargs['method'])

    moving.transform(result.transformation)
    moving.translate(center)

    A = np.eye(4)
    A[0:3,3] = -center
    C = np.eye(4)
    C[0:3,3] = center
    # print(A)
    # print(result.transformation)
    # print(C)
    print(np.matmul(np.matmul(C, result.transformation), A))
    # print(np.matmul(C, result.transformation))

    # We must read in Fortran order, which is how Open3D treats the points!
    pts = np.asarray(moving.points, order='F')
    outs['X'] = pts[:, 0]
    outs['Y'] = pts[:, 1]
    outs['Z'] = pts[:, 2]

    return True
