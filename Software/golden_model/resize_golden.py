"""
resize_golden.py
FPGA fixed-point resize golden model.

Specification:
- Output: 960x720 by default
- Preserve aspect ratio
- Centered black letterbox padding
- Bilinear interpolation
- 8 fractional bits for interpolation weights
- Round-to-nearest
- Edge-clamped borders
- No cv2.resize()
"""

from PIL import Image
import numpy as np

OUT_W = 960
OUT_H = 720
FP_BITS = 8
FP_ONE = 1 << FP_BITS          # 256
ROUND_OFFSET = 1 << 15        # 32768


def calculate_resize_parameters(in_w, in_h, out_w=OUT_W, out_h=OUT_H):
    if in_w <= 0 or in_h <= 0:
        raise ValueError("Input dimensions must be positive.")

    scale = min(out_w / in_w, out_h / in_h)

    scaled_w = max(1, min(out_w, int(round(in_w * scale))))
    scaled_h = max(1, min(out_h, int(round(in_h * scale))))

    total_pad_x = out_w - scaled_w
    total_pad_y = out_h - scaled_h

    pad_left = total_pad_x // 2
    pad_top = total_pad_y // 2

    return {
        "scale": scale,
        "scaled_w": scaled_w,
        "scaled_h": scaled_h,
        "pad_left": pad_left,
        "pad_right": total_pad_x - pad_left,
        "pad_top": pad_top,
        "pad_bottom": total_pad_y - pad_top,
    }


def fixed_bilinear_pixel(p00, p10, p01, p11, a, b):
    a_fixed = max(0, min(FP_ONE, int(round(a * FP_ONE))))
    b_fixed = max(0, min(FP_ONE, int(round(b * FP_ONE))))

    A = FP_ONE - a_fixed
    B = FP_ONE - b_fixed

    accum = (
        int(p00) * A * B
        + int(p10) * a_fixed * B
        + int(p01) * A * b_fixed
        + int(p11) * a_fixed * b_fixed
    )

    value = (accum + ROUND_OFFSET) >> 16
    return max(0, min(255, value))


def resize_fixed(image, out_w=OUT_W, out_h=OUT_H):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 image.")

    in_h, in_w, channels = image.shape
    params = calculate_resize_parameters(in_w, in_h, out_w, out_h)

    scale = params["scale"]
    scaled_w = params["scaled_w"]
    scaled_h = params["scaled_h"]
    pad_left = params["pad_left"]
    pad_top = params["pad_top"]

    image = image.astype(np.uint8)
    output = np.zeros((out_h, out_w, channels), dtype=np.uint8)

    for y_out in range(out_h):
        if y_out < pad_top or y_out >= pad_top + scaled_h:
            continue

        y_scaled = y_out - pad_top
        y_src = y_scaled / scale

        y0 = int(np.floor(y_src))
        y1 = y0 + 1
        b = y_src - y0

        y0 = max(0, min(in_h - 1, y0))
        y1 = max(0, min(in_h - 1, y1))

        for x_out in range(out_w):
            if x_out < pad_left or x_out >= pad_left + scaled_w:
                continue

            x_scaled = x_out - pad_left
            x_src = x_scaled / scale

            x0 = int(np.floor(x_src))
            x1 = x0 + 1
            a = x_src - x0

            x0 = max(0, min(in_w - 1, x0))
            x1 = max(0, min(in_w - 1, x1))

            for c in range(channels):
                output[y_out, x_out, c] = fixed_bilinear_pixel(
                    image[y0, x0, c],
                    image[y0, x1, c],
                    image[y1, x0, c],
                    image[y1, x1, c],
                    a, b
                )

    return output, params


def resize_file(input_path, output_path, out_w=OUT_W, out_h=OUT_H):
    image = np.array(Image.open(input_path).convert("RGB"))
    output, params = resize_fixed(image, out_w, out_h)
    Image.fromarray(output).save(output_path)

    print(f"Input : {image.shape[1]}x{image.shape[0]}")
    print(f"Output: {out_w}x{out_h}")
    print(f"Scale : {params['scale']}")
    print(f"Scaled: {params['scaled_w']}x{params['scaled_h']}")
    print(
        f"Padding: left={params['pad_left']}, right={params['pad_right']}, "
        f"top={params['pad_top']}, bottom={params['pad_bottom']}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--width", type=int, default=OUT_W)
    parser.add_argument("--height", type=int, default=OUT_H)
    args = parser.parse_args()

    resize_file(args.input, args.output, args.width, args.height)
