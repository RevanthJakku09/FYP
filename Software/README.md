# FPGA-Based Image Preprocessing for YuNet Face Detection

## 1. Project Overview

This project investigates implementing reusable image preprocessing operations in FPGA hardware and evaluating their usefulness in a face-detection pipeline using OpenCV YuNet.

The software phase establishes a baseline, evaluates preprocessing alternatives, selects a fixed output resolution, and develops a Python reference model for the resize operation. This reference will later be used to verify the corresponding Verilog implementation.

**Current selected configuration:**
- Application: YuNet face detection
- Selected preprocessing operation: Image resizing with aspect-ratio preservation and letterbox padding
- Target output resolution: 960 × 720
- Reference implementation: Python resize golden model
- Hardware implementation: Planned; not started

## 2. Project Objectives

1. Establish a working YuNet face-detection software baseline.
2. Evaluate whether additional preprocessing operations improve detection quality.
3. Compare fixed output resolutions using detection metrics and processing time.
4. Select a suitable preprocessing configuration for hardware implementation.
5. Develop a Python golden model that defines the expected resize output.
6. Later implement the selected resize operation in Verilog and compare hardware output against the Python reference.
7. Eventually evaluate FPGA resource usage, timing, and performance.

## 3. Software Project Structure

The software project contains the following main directories and files. The exact list of scripts and assets may change as the project develops.

```text
FYP/
├── README.md
└── FaceDetection_Software/
    ├── .gitignore
    └── Software/
        ├── datasets/
        ├── golden_model/
        │   ├── resize_golden.py
        │   └── batch_test_resize.py
        ├── models/
        ├── output/
        ├── test_images/
        ├── .venv/                 # Local environment; not committed
        ├── requirements.txt
        ├── preprocessing.py
        ├── fpga_blocks.py
        ├── run_yunet.py
        ├── run_yunet_complete.py
        ├── run_yunet_selective.py
        └── run_yunet_final.py
```

Some scripts may exist only in the local working tree or may have different responsibilities than their filenames suggest. Consult each script's source and command-line help for its exact behavior.

## 4. Development Environment

The software was developed using Python on Windows with PowerShell and a project-specific virtual environment.

### Main dependencies

- **OpenCV (`opencv-python`)** — image processing, resizing, and YuNet integration.
- **NumPy** — image arrays and numerical calculations.
- **Pillow** — image loading and processing in the resize golden model.
- **Python standard library** — utilities such as `argparse`, `pathlib`, `os`, and `time`.

The exact dependencies required by each script are defined by the project's `requirements.txt`.

### Setup commands

Open PowerShell and navigate to the software directory:

```powershell
cd "D:\FYP_SOFTWARE\FaceDetection_Software\Software"
```

Activate the virtual environment if it already exists there:

```powershell
.\.venv\Scripts\Activate.ps1
```

Check Python and pip:

```powershell
python --version
python -m pip --version
```

Install the project's declared dependencies:

```powershell
python -m pip install -r requirements.txt
```

If the main external libraries are missing, install them with:

```powershell
python -m pip install opencv-python numpy Pillow
```

Verify the imports:

```powershell
python -c "import cv2, numpy; from PIL import Image; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('Pillow: OK')"
```

**Note:** The YuNet model file is separate from the Python packages. Ensure the model exists at the path expected by the selected runner.

## 5. YuNet Baseline

The baseline is the reference for evaluating preprocessing changes.

The baseline workflow reads input images, applies the original resizing strategy, runs YuNet, and evaluates detections. The exact metrics and timing depend on the evaluation script and its configuration.

The principal baseline runner is:

`run_yunet_complete.py`

Inspect its available options:

```powershell
python run_yunet_complete.py --help
```

Run it using the arguments supported by the script. If it supports default arguments, it may be launched with:

```powershell
python run_yunet_complete.py
```

The baseline is important because preprocessing should be judged by measured changes in detection quality and computational cost, not merely by whether the operation can be implemented.

## 6. Preprocessing Experiments

The software experiments compared resizing alone with pipelines that also included colour conversion and Gaussian filtering.

### Operations considered

**Resize:** Changes image dimensions to a selected target size.

**Colour conversion:** Changes the image colour representation. Its effect depends on the chosen conversion and the detector's expected input format.

**Gaussian filtering:** Smooths the image and can reduce some high-frequency noise, but introduces additional computation and may remove useful detail.

### Recorded experiment results

| Pipeline | Detection F1 | Image F1 | Reported processing time |
|---|---:|---:|---:|
| Resize only | 0.7629 | 0.9713 | 1.07 ms/image |
| Resize + colour | 0.5462 | 0.9179 | 2.86 ms/image |
| Resize + Gaussian | 0.7551 | 0.9683 | 70.73 ms/image |
| Resize + colour + Gaussian | 0.5473 | 0.9214 | 68.54 ms/image |

These are the previously recorded experiment results. The exact definition of image F1 and the timing scope should be checked against the evaluation code before formal publication.

### Conclusion

Resize only produced the highest recorded detection F1 among these tested preprocessing pipelines. Colour conversion reduced the recorded detection F1, while Gaussian filtering added substantial processing time without improving the detection F1.

Therefore, the selected application pipeline uses **resize only**. Other operations may still be useful as reusable blocks for future applications, but they are not selected for the current YuNet pipeline.

## 7. Resolution Experiments

Different fixed output resolutions were compared to evaluate the trade-off between detection quality and computation.

### Recorded results

| Output resolution | Precision | Recall | Detection F1 | YuNet time | Reported FPS |
|---|---:|---:|---:|---:|---:|
| 320 × 240 | 0.8637 | 0.2065 | 0.3333 | 7.16 ms | 127.04 |
| 640 × 480 | 0.8879 | 0.4882 | 0.6300 | 21.76 ms | 43.62 |
| 960 × 720 | 0.8868 | 0.6320 | 0.7381 | 49.67 ms | 19.01 |

The variable-size baseline recorded a detection F1 of 0.7629. Therefore, 960 × 720 performed best among the fixed resolutions tested, but did not exceed the variable-size baseline's recorded F1.

### Metrics

- **Precision:** Fraction of predicted detections that are correct.
- **Recall:** Fraction of actual faces that are detected.
- **Detection F1:** Harmonic mean of precision and recall.
- **YuNet time:** Detector execution time as measured by the experiment.
- **FPS:** Frames per second as reported by the experiment's timing method.

### Resolution decision

The selected fixed target is **960 × 720**. This choice prioritizes detection quality among the fixed resolutions tested; it is not a claim that the resolution is optimal for every dataset or hardware platform.

## 8. Resize Golden Model

The `golden_model/` directory contains a Python reference implementation of the resize operation.

A golden model is a trusted reference used to verify another implementation. In this project, it will define the expected pixel output for the future Verilog resize block.

The main reference file is:

`golden_model/resize_golden.py`

The batch test is:

`golden_model/batch_test_resize.py`

### Intended resize behavior

The golden model implements:

- Target dimensions of 960 × 720.
- Aspect-ratio-preserving scaling.
- Centered black letterbox padding.
- Bilinear interpolation.
- Fixed-point interpolation weights using eight fractional bits.
- Round-to-nearest output calculation.
- Edge clamping at image borders.

The precise coordinate mapping, integer arithmetic, and rounding behavior must be taken from the actual implementation when developing the Verilog module.

### Letterbox scaling

Let the input dimensions be \(W_i \times H_i\) and the target dimensions be \(W_t \times H_t\).

The scale factor is:

\[
s=\min\left(\frac{W_t}{W_i},\frac{H_t}{H_i}\right)
\]

The resized dimensions are approximately:

\[
W_r=\operatorname{round}(sW_i),\qquad
H_r=\operatorname{round}(sH_i)
\]

The remaining horizontal and vertical spaces are used for padding. Padding is distributed around the resized image to center it in the target frame. Any odd leftover pixel is assigned according to the implementation's padding rules.

### Bilinear interpolation

Bilinear interpolation estimates an output pixel from four neighbouring source pixels:

\[
\begin{aligned}
P(x,y)={}&(1-a)(1-b)P_{00}\\
&+a(1-b)P_{10}\\
&+(1-a)bP_{01}\\
&+abP_{11}
\end{aligned}
\]

Here, \(a\) and \(b\) represent the fractional horizontal and vertical positions. The four source pixels are \(P_{00}\), \(P_{10}\), \(P_{01}\), and \(P_{11}\).

Using fixed-point weights makes the arithmetic more suitable for Verilog than a floating-point implementation, but matching the Python reference requires identical coordinate mapping, quantization, border handling, and rounding.

Python's `round()` uses ties-to-even behavior for exact halfway cases. This must be considered when reproducing the arithmetic in hardware.

### Golden-model batch testing

The batch test applies the Python resize operation to multiple images and checks output properties such as dimensions and pixel-value ranges.

The basic and batch tests were reported as passing in the development work. These tests establish software-level behavior for the tested inputs; they do not yet establish pixel-exact equivalence with Verilog.

## 9. Final Software Runner

The intended final runner is:

`run_yunet_final.py`

Its selected application pipeline is:

1. Read an input image.
2. Resize it while preserving its aspect ratio.
3. Add centered black padding to produce a 960 × 720 image.
4. Run YuNet on the processed image.
5. Generate detection results and performance information according to the script's implementation.

The runner was created using OpenCV's `cv2.INTER_LINEAR` resizing. OpenCV resizing is not automatically pixel-identical to the custom golden model. The coordinate mapping, interpolation weights, rounding, and border behavior must be reconciled before claiming exact agreement.

Inspect available command-line options:

```powershell
python run_yunet_final.py --help
```

Run the script with its supported arguments. If it supports defaults, use:

```powershell
python run_yunet_final.py
```

The script's actual source and runtime output determine which files, metrics, and detections are produced.

## 10. Useful Verification Commands

List Python files:

```powershell
Get-ChildItem -Path . -Filter "*.py" -File -Recurse |
    Select-Object FullName
```

Inspect installed packages:

```powershell
python -m pip list
```

Check installed package dependency consistency:

```powershell
python -m pip check
```

Compile-check Python scripts:

```powershell
python -m compileall -q .
```

Compilation checks syntax but does not execute the scripts or verify numerical correctness.

Inspect golden-model files:

```powershell
Get-ChildItem ".\golden_model" -File
```

Inspect generated output files:

```powershell
Get-ChildItem ".\output" -File -Recurse |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 10 Name, LastWriteTime, Length
```

## 11. Current Status

| Item | Status |
|---|---|
| YuNet baseline | Established |
| Preprocessing comparison | Results recorded |
| Fixed-resolution comparison | Results recorded |
| Selected preprocessing pipeline | Resize only |
| Selected target resolution | 960 × 720 |
| Python resize golden model | Created |
| Golden-model basic and batch tests | Reported passing |
| Final software runner | Created; runtime results should be retained with project records |
| Verilog resize implementation | Not started |
| Verilog-versus-Python pixel comparison | Not started |
| FPGA synthesis and resource analysis | Not started |

## 12. Future Hardware Verification Plan

The planned next phase is to implement the selected resize operation in Verilog and verify it against the Python golden model.

The verification plan includes:

1. Define the hardware module interface.
2. Implement the resize algorithm using matching fixed-point arithmetic.
3. Create a testbench with representative input images.
4. Compare the hardware output with the Python reference.
5. Check output dimensions, pixel mismatch count, and maximum absolute pixel error.
6. Investigate any differences in interpolation, rounding, padding, or edge handling.
7. Synthesize the verified design and evaluate FPGA resource usage and timing.

The goal is to demonstrate correct preprocessing behavior before integrating the hardware block into a complete image-processing system.

## 13. Repository and File Management

The project is maintained in the existing GitHub repository:

https://github.com/RevanthJakku09/FYP

The software is being added first. Hardware files will be added later to the same repository when hardware implementation begins.

The Python virtual environment, Python caches, and unnecessary generated outputs should not be committed. Keep relevant source files, dependency declarations, reference-model code, test assets, and documented experimental results.

## 14. Final Summary

The software phase established a YuNet baseline, evaluated alternative preprocessing pipelines, compared fixed output resolutions, and developed a Python golden model for resizing images to 960 × 720.

Resize only was selected because it provided the highest recorded detection F1 among the tested preprocessing pipelines. The fixed 960 × 720 configuration performed best among the tested fixed resolutions, although the variable-size baseline had a slightly higher recorded detection F1.

The next phase will implement the resize operation in Verilog and verify it against the Python reference before evaluating FPGA performance and resource utilization.