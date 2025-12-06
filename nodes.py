import torch
import nodes
import comfy.model_management
from nodes import MAX_RESOLUTION


class ShakaWanKeyframes:
    """
    Accepts a batch of images and specific frame indices.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "frame_indices": ("STRING", {"default": "1, 81", "multiline": False}),
                "frame_weights": ("STRING", {"default": "1.0, 1.0", "multiline": False}),
                "shaka_mode": ("BOOLEAN", {"default": True, "tooltip": "Shaka Mode ensures compatibility with Wan's conditioning format. Ensures frames stick to a 4-frame temporal grid"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT",)
    RETURN_NAMES = ("images", "masks", "width", "height", "length",)
    FUNCTION = "execute"
    CATEGORY = "ShakaNodes/Wan"

    def execute(self, images, width, height, length, frame_indices, frame_weights, shaka_mode):
        self._sanity_check(images, width, height, length, frame_indices, frame_weights, shaka_mode)
        return self._execute(images, width, height, length, frame_indices, frame_weights, shaka_mode)

    def _sanity_check(self, images, width, height, length, frame_indices, frame_weights, shaka_mode):
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError("🌊 ShakaWanKeyframes: Width and Height must be multiples of 16.")
        if (length - 1) % 4 != 0:
            raise ValueError("🌊 ShakaWanKeyframes: Length must be 1 plus a multiple of 4 (e.g., 1,5,9,13...).")
        if images.shape[0] < 1:
            raise ValueError("🌊 ShakaWanKeyframes: At least one image must be provided.")
        if images.shape[0] > (length - 1) // 4:
            raise ValueError("🌊 ShakaWanKeyframes: Too many images for the specified length.")
        try:
            raw_indices = [int(idx.strip()) for idx in frame_indices.split(",")]
        except ValueError:
            raise ValueError("🌊 ShakaWanKeyframes: Frame indices must be comma-separated integers.")
        if images.shape[0] != len(raw_indices):
            raise ValueError(f"🌊 ShakaWanKeyframes: Input mismatch! {images.shape[0]} images vs {len(raw_indices)} indices.")
        if any(idx < 1 or idx > length for idx in raw_indices):
            raise ValueError(f"🌊 ShakaWanKeyframes: Frame indices must be between 1 and {length}.")
        try:
            raw_weights = [float(weight.strip()) for weight in frame_weights.split(",")]
        except ValueError:
            raise ValueError("🌊 ShakaWanKeyframes: Frame weights must be comma-separated floats.")
        if images.shape[0] != len(raw_weights):
            raise ValueError(f"🌊 ShakaWanKeyframes: Input mismatch! {images.shape[0]} images vs {len(raw_weights)} indices.")
        if any(weight < 0.0 or weight > 1.0 for weight in raw_weights):
            raise ValueError("🌊 ShakaWanKeyframes: Frame weights must be between 0.0 and 1.0.")
        if shaka_mode:
            snapped_indices = [((idx - 1) // 4) * 4 + 1 for idx in raw_indices]
            if len(set(snapped_indices)) < len(snapped_indices):
                raise ValueError("🌊 ShakaWanKeyframes: In Shaka Mode, frame indices must not conflict within the same 4-frame group.")

    def _execute(self, images, width, height, length, frame_indices, frame_weights, shaka_mode):
        device = comfy.model_management.intermediate_device()

        # Prep Tensors
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask = torch.ones((length, height, width), device=device)

        # Process frame indices and weights
        raw_indices = [int(idx.strip()) for idx in frame_indices.split(",")]
        weights = [float(weight.strip()) for weight in frame_weights.split(",")]

        indices = [idx - 1 for idx in raw_indices]
        if shaka_mode:
            # Snap indices to nearest lower multiple of 4
            indices = [((idx - 1) // 4 ) * 4 for idx in raw_indices]

        for i, idx in enumerate(indices):
            frame = comfy.utils.common_upscale(
                images[i].unsqueeze(0).movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)[0]

            image[idx] = frame
            mask[idx] = 1 - weights[i]

        print(f"🌊 ShakaWanKeyframes: Prepped Images and Masks for VACE Wan")

        return (image, mask, width, height, length,)


# from https://github.com/pythongosssss/ComfyUI-Custom-Scripts
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class ShakaTensorDebug:
    """
    Connect this to any output to see what is actually inside!
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (AnyType("*"), {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "ShakaNodes/Utils"

    def execute(self, tensor):
        print(f"\n🌊 --- SHAKA DEBUG: ---")
        if isinstance(tensor, dict) and "samples" in tensor:
            lat = tensor["samples"]
            print(f"Type: LATENT")
            print(f"Shape: {lat.shape} (Batch, Channel, Time, Height, Width)")
        elif isinstance(tensor, list) and isinstance(tensor[0], tuple):
            print(f"Type: CONDITIONING")
            print(f"Batch Size: {len(tensor)}")
            try:
                cond_tensor = tensor[0][0]
                params = tensor[0][1]
                print(f"Main Tensor Shape: {cond_tensor.shape}")
                if "concat_mask" in params:
                    mask = params["concat_mask"]
                    print(f"Found Mask! Shape: {mask.shape}")
                    try:
                        # Grab a tiny slice to see if it's 0.0 (Unmasked) or 1.0 (Masked)
                        # Squeeze dimensions to get to values
                        flat_mask = mask.mean(dim=(3,4)).flatten()[:12]
                        print(f"Mask Sample (First block): {flat_mask}")
                    except:
                        print("Could not flatten mask.")
            except:
                print("Could not parse conditioning structure.")
        else:
            print(f"Type: {type(tensor)}")
            try:
                print(f"Value: {tensor}")
            except:
                pass

        print("🌊 ----------------------------\n")
        return {}