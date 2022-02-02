# script.py

import json
import open3d as o3d
import pdal
import sys

def main(argv):
    moving_filename = argv[0]
    fixed_filename = argv[1]
    aligned_filename = argv[2]

    # Read the fixed point cloud so that we can later get it's center coords
    f = pdal.Reader(fixed_filename).pipeline()
    f.execute()

    # Create the reader and pipeline in one step
    p = pdal.Reader(moving_filename).pipeline()

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
    
    return 0

if __name__ == "__main__":
    print(sys.argv)
    sys.exit(main(sys.argv[1:]))
