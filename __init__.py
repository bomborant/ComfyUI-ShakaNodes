from .nodes import ShakaWanKeyframes, ShakaTensorDebug

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ShakaWanKeyframes": ShakaWanKeyframes,
    "ShakaTensorDebug": ShakaTensorDebug,
}

# A dictionary that contains the friendly/human-readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShakaWanKeyframes": "🤙 Shaka Wan Keyframes",
    "ShakaTensorDebug": "🤙 Shaka Tensor Debugger",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]