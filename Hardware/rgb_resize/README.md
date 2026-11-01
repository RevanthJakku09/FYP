# FPGA-Based RGB Image Resizing and Padding Using Verilog HDL

## Overview

This project implements an RGB image resizing pipeline in Verilog HDL.
It maps an input image onto a configurable output canvas, preserves the
image aspect ratio, and uses **bilinear interpolation** to calculate
resized RGB pixel values. If the resized image does not fill the output
canvas, the remaining area is filled with black pixels.

The supplied testbench is configured for:

-   **Input resolution:** 640 × 480
-   **Output resolution:** 960 × 720
-   **Pixel format:** 24-bit RGB (8 bits per channel)
-   **Interpolation:** Bilinear interpolation
-   **Output scan order:** Row by row

Both resolutions have a 4:3 aspect ratio, so this particular
configuration fills the output canvas without adding padding. The RTL
also calculates centered padding for other input/output aspect-ratio
combinations.

## Features

-   Parameterized input and output image dimensions.
-   Aspect-ratio-preserving resize-dimension calculation.
-   Centered horizontal and vertical padding when required.
-   24-bit RGB pixel processing (`R[23:16]`, `G[15:8]`, `B[7:0]`).
-   Q8 fixed-point source-coordinate mapping.
-   Four-neighbor pixel access from frame memory.
-   Bilinear interpolation performed independently on the red, green,
    and blue channels.
-   Synchronous frame-memory read and write operations.
-   Black output for padding regions.
-   Testbench generation of a deterministic RGB input pattern and export
    of output pixels to a text file.

## Project Architecture

The design is divided into the following modules:

  ---------------------------------------------------------------------------------
  File / Module                                 Responsibility
  --------------------------------------------- -----------------------------------
  `rgb_memory_padding_demo.v` /                 Top-level resize and padding
  `rgb_memory_padding_demo`                     datapath. Calculates scaled
                                                dimensions and padding, connects
                                                coordinate mapping, frame memory,
                                                and interpolation.

  `resize_coords.v` / `resize_coords`           Maps output coordinates to
                                                source-image coordinates and
                                                calculates the four neighboring
                                                pixel positions and Q8
                                                interpolation fractions.

  `rgb_frame_memory.v` / `rgb_frame_memory`     Stores 24-bit RGB pixels and
                                                provides four synchronous read
                                                ports for the neighboring pixels.

  `resize.v` / `resize`                         Calculates the bilinear
                                                interpolated value for each RGB
                                                channel.

  `rgb_memory_resize_640x480_to_960x720_tb.v`   Simulation testbench. Loads a
                                                generated 640 × 480 frame, scans
                                                the 960 × 720 output, checks
                                                validity/unknown values, and writes
                                                pixel data to a text file.
  ---------------------------------------------------------------------------------

### Processing Flow

1.  **Input frame loading:** The testbench writes each 24-bit RGB pixel
    into frame memory using a row-major address.
2.  **Output coordinate scan:** The testbench supplies each output
    `(x, y)` coordinate.
3.  **Aspect-ratio and padding calculation:** The top-level module
    calculates the resized image dimensions and centers the image in the
    output canvas.
4.  **Coordinate mapping:** `resize_coords` maps the output position to
    a source position using Q8 fixed-point arithmetic. It determines
    `x0`, `x1`, `y0`, `y1`, and interpolation fractions `a` and `b`.
5.  **Neighbor pixel reading:** `rgb_frame_memory` reads the four
    surrounding source pixels: `p00`, `p10`, `p01`, and `p11`.
6.  **Interpolation:** `resize` applies bilinear interpolation
    independently to the R, G, and B channels.
7.  **Output generation:** The resized pixel is produced inside the
    resized-image area. Pixels outside that area are set to black
    (`24'h000000`).

## Bilinear Interpolation

For each color channel, the four neighboring source pixels are combined
using horizontal and vertical interpolation fractions.

Let `a` and `b` be the fractional coordinates, represented in Q8 format,
with a scale of 256. The weights are:

-   `w00 = (256 - a) × (256 - b)`
-   `w10 = a × (256 - b)`
-   `w01 = (256 - a) × b`
-   `w11 = a × b`

The interpolated channel value is calculated as:

`P = (p00*w00 + p10*w10 + p01*w01 + p11*w11) / 65536`

The RTL uses a rounding offset before the fixed-point result is
shifted/scaled back to an 8-bit channel value.

## Testbench Details

The testbench creates a deterministic RGB pattern from each input
coordinate:

-   `R = (x + y) mod 256`
-   `G = (2*x + y) mod 256`
-   `B = (x + 3*y) mod 256`

It writes all **307,200 input pixels** to frame memory, then scans all
**691,200 output coordinates**.

For every output coordinate, the testbench checks that:

-   `pixel_valid` is asserted.
-   `pixel_out` does not contain unknown (`X`) bits.

It exports each output pixel in this text format:

`x y RRGGBB`

The testbench reports the number of checked output pixels and RTL
errors, and prints a pass/fail message.

**Verification note:** These checks are validity and unknown-value
checks. The supplied testbench does not compare every output RGB value
against a separately calculated golden reference image, so a passing
result alone does not prove pixel-by-pixel interpolation accuracy.

## Running the Simulation in Vivado

1.  Open **Xilinx Vivado**.
2.  Create a new RTL project, or open your existing project.
3.  Add the four design files as **Design Sources**:
    -   `rgb_memory_padding_demo.v`
    -   `resize_coords.v`
    -   `rgb_frame_memory.v`
    -   `resize.v`
4.  Add `rgb_memory_resize_640x480_to_960x720_tb.v` as a **Simulation
    Source**.
5.  Set `rgb_memory_resize_640x480_to_960x720_tb` as the simulation top
    module.
6.  Run **Behavioral Simulation**.
7.  Check the Tcl Console for the test progress and final pass/fail
    message.
8.  Check the output text file generated by the testbench.

The testbench currently contains a machine-specific absolute Windows
output path. Change the `$fopen` path to a location that exists on your
computer, or use a relative filename such as `rgb_960x720_output.txt` in
the simulator's working directory.

## Important Source-File Checks Before Running

The uploaded `resize.v` copy appears to have lost Verilog syntax during
export or copying. In particular, its vector ranges, channel slices,
comments, and arithmetic operators are not valid as displayed. Check
that the actual source file uses the intended declarations and operators
before adding it to Vivado. For a 24-bit RGB interface, the expected
vector range is `[23:0]`; the channel ranges are `[23:16]`, `[15:8]`,
and `[7:0]`. The interpolation expressions also require explicit
multiplication and shift/division operators.

Also ensure that the design files are saved with the `.v` extension and
that all module names match the instantiated module names.

## Repository Structure

A suggested Git repository layout is:

``` text
RGB-Image-Resizing/
├── README.md
├── RTL/
│   ├── rgb_memory_padding_demo.v
│   ├── resize_coords.v
│   ├── rgb_frame_memory.v
│   └── resize.v
└── Simulation/
    └── rgb_memory_resize_640x480_to_960x720_tb.v
```

You can use a different folder structure; update the Vivado source paths
accordingly.

## Technologies

-   Verilog HDL
-   Xilinx Vivado
-   RTL Design
-   Digital Image Processing
-   FPGA-based image processing
-   Bilinear Interpolation
-   RGB Frame Memory
-   Fixed-Point Arithmetic

## Future Scope

-   Add a pixel-by-pixel comparison against a software golden model.
-   Test additional input and output resolutions.
-   Integrate an image/video input interface and a display output
    interface.
-   Evaluate FPGA resource utilization and maximum operating frequency
    after synthesis and implementation.

## Author

Add your name, university, project details, and contact or GitHub
profile here.
