import torch
import typing
from .format import LUTFormat

import os

def _infer_lut_file_format(ext:str) -> LUTFormat:
    ext = ext.strip().lower()
    if   ext == '.cube':     return LUTFormat.CUBE_3D
    else: return LUTFormat.UNKNOWN

def load_lut(fp:str, lut_format:LUTFormat=LUTFormat.UNKNOWN) -> torch.Tensor:
    """
    Load LUT from file.

    :param fp: File path to load from.
    :param lut_format: Format of the LUT.
    :return: lattice as PyTorch tensor
    """
    fp = os.path.realpath(fp)
    fn, fnext = os.path.splitext(fp)
    if lut_format == LUTFormat.UNKNOWN: lut_format = _infer_lut_file_format(fnext)
    assert lut_format > LUTFormat.UNKNOWN, "Unrecognized LUT format"
    lattice = None
    if lut_format == LUTFormat.CUBE_3D:
        with open(fp, 'r') as file:
            # Read lines and filter out comments
            lines = [line.strip() for line in file.readlines() if len(line) > 0 and "#" not in line]

            # Find the line indicating the start of the LUT data
            lut_start_index = next((i for i, line in enumerate(lines) if line.startswith('LUT_3D_SIZE')), None)

            if lut_start_index is None: raise ValueError("LUT_3D_SIZE indicator not found in the .cube file.")

            # Extract LUT data
            lut_data = [list(map(float, line.split())) for line in lines[lut_start_index + 1:] \
                if len(line) > 0 and len(line.split()) == 3 and "#" not in line]    

        # Convert the LUT data to a PyTorch tensor
        lut_size = int(lines[lut_start_index].split()[1])

        lattice = torch.tensor(lut_data).view(lut_size, lut_size, lut_size, 3).permute(2,1,0,3)
    else:
        raise Exception("Unsupported LUT format")
    return lattice

def save_lut(lattice:torch.Tensor, fp:str, lut_format:LUTFormat=LUTFormat.UNKNOWN) -> bool:
    """
    Save LUT to a file.

    .. warning:: 
        This will overwrite existing files.

    :param lattice: PyTorch tensor representing the LUT.
    :param fp: File path for saving the LUT.
    :param lut_format: Format of the LUT.
    :return: True if successful
    """
    fp = os.path.realpath(fp)
    fn, fnext = os.path.splitext(fp)

    if lut_format == LUTFormat.UNKNOWN: lut_format = _infer_lut_file_format(fnext)

    if lut_format == LUTFormat.CUBE_3D:
        # Convert the torch tensor to a list of lists
        lut_data_list = lattice.permute(2,1,0,3).reshape(-1, 3).tolist()

        # Write the LUT data to the file
        with open(fp, 'w') as file:
            file.write(f"LUT_3D_SIZE {lattice.size(0)}\n")
            for entry in lut_data_list:
                file.write(" ".join(map(str, entry)) + "\n")

    else:
        raise Exception("Unsupported LUT format")

    return True

def _generate_linear_cube_lut(size: int):
    """
    Generate a baseline linear cube LUT.

    :param size: Size of the cube LUT (e.g., 33 for a 33x33x33 cube).
    :return: Torch tensor representing the linear cube LUT.
    """
    linear_values = torch.linspace(0.0, 1.0, size)
    grid = torch.meshgrid(linear_values, linear_values, linear_values, indexing="xy")
    lut_data = torch.stack(grid, dim=-1).permute(1,0,2,3) # TODO: WTF is even going on here?
    return lut_data