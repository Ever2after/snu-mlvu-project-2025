import av
import numpy as np
import base64
import re

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def extract_assistant_response(response):
    """
    Extracts the answer portion following 'ASSISTANT:' in a response string.

    Parameters
    ----------
    response : str
        The full response string, e.g. '... ASSISTANT: The answer content'

    Returns
    -------
    str
        The text after 'ASSISTANT:', trimmed of leading/trailing whitespace.
        Returns an empty string if no match is found.
    """
    # Search for 'ASSISTANT:' followed by any characters (including newlines)
    match = re.search(r"ASSISTANT:\s*(.+)", response, flags=re.DOTALL)
    # If found, return the captured group; otherwise return empty
    return match.group(1).strip() if match else ""
