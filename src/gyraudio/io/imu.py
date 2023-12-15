from pathlib import Path
import gpmf
import numpy as np
from gyraudio.properties import GYRO_KEY, ACCL_KEY
def extract_imu_blocks(stream, key=GYRO_KEY):
    """ Extract imu data blocks from binary stream

    This is a generator on lists `KVLItem` objects. In
    the GPMF stream, imu data comes into blocks of several
    different data items. For each of these blocks we return a list.

    Parameters
    ----------
    stream: bytes
        The raw GPMF binary stream

    Returns
    -------
    imu_items_generator: generator
        Generator of lists of `KVLItem` objects
    """
    for s in gpmf.parse.filter_klv(stream, "STRM"):
        content = []
        is_imu = False
        for elt in s.value:
            content.append(elt)
            if elt.key == key:
                is_imu = True
        if is_imu:
            yield content


def parse_imu_block(imu_block, key=GYRO_KEY):
    """Turn imu data blocks into `imuData` objects

    Parameters
    ----------
    imu_block: list of KVLItem
        A list of KVLItem corresponding to a imu data block.

    Returns
    -------
    imu_data: imuData
        A imuData object holding the imu information of a block.
    """
    block_dict = {
        s.key: s for s in imu_block
    }

    imu_data = block_dict[key].value * 1.0 / block_dict["SCAL"].value
    return {
        "timestamp": block_dict["TSMP"],
        key: imu_data,
    }


def get_imu_data(pth: Path) -> np.array:
    """Extract imu data from GPMF metadata from a Gopro video file

    Parameters
    ----------
    path: str
        Path to the GPMF file

    Returns
    -------
    imu_data: list of imuData
        List of imuData objects holding the imu information of the file.
    """
    stream = gpmf.io.extract_gpmf_stream(pth)
    imu_data_dict = {}
    for key in [GYRO_KEY, ACCL_KEY]:
        imu_blocks = extract_imu_blocks(stream, key=key)
        imu_data = [parse_imu_block(imu_block, key=key) for imu_block in imu_blocks]
        imu_data_dict[key] = np.vstack([np.array(imu[key]) for imu in imu_data])
    return imu_data_dict[GYRO_KEY], imu_data_dict[ACCL_KEY]
