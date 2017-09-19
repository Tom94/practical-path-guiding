## Practical Path Guiding for Efficient Light-Transport Simulation

This repository contains the authors' implementation of the guided unidirectional path tracer of the research paper ["Practical Path Guiding for Efficient Light-Transport Simulation" [Müller et al. 2017]](https://tom94.net). It also includes a visualization tool for the SD-Trees learned by the guided path tracer. The guided path tracer has been implemented in the [Mitsuba Physically Based Renderer](http://mitsuba-renderer.org) and the visualization tool with the [nanogui](https://github.com/wjakob/nanogui) library.

## Implementation

- The guided path tracer is implemented as the `GuidedPathTracer` Mitsuba integrator.
- The visualization tool is implemented as a standalone program built on nanogui.

### Modifications to Mitsuba

- `BlockedRenderProcess` (*renderproc.cpp, renderproc.h*)
  - Allowed retrieving the total amount of blocks.
  - Disabled automatic progress bookkeeping.
- `GuidedPathTracer` (*guided_path.cpp*)
  - Added the guided path tracer implementing [Müller et al. 2017].
- `ImageBlock` (*imageblock.h*)
  - Allowed querying the reconstruction filter.
- `MainWindow` (*mainwindow.cpp*)
  - Removed warning about orphaned rectangular work units (occured when multiple threads write into spatially overlapping blocks at the same time).
- General
  - Removed the Irawan BSDF to make mitsuba compile under newer GCC versions.
  - Always enabled the ADT_WORKAROUND in `ply_parser.hpp` to make mitsuba compile under newer GCC versions.

### Modifications to nanogui

- `ImageView` (*imageview.cpp, imageview.h*)
  - Changed the shader to display a false-color visualization of a given high-dynamic-range image.
- General
  - Removed `noexcept` qualifiers to make nanogui compile under Visual Studio 2013.
  - Removed `constexpr` qualifiers to make nanogui compile under Visual Studio 2013.

## Compilation

### Mitsuba

To compile the Mitsuba code base, follow the instructions from the [Mitsuba documentation](http://mitsuba-renderer.org/docs.html). Since our new code uses C++11 features, a slightly more recent compiler than reported in the mitsuba documentation may be required.

We tested our Mitsuba code on
- Windows (Visual Studio 2013 Win64)
- Linux (GCC 6.3.1)

### Visualization Tool

The visualization tool, found in the *visualizer* subfolder, uses the [CMake](https://cmake.org/) build system. Simply invoke the CMake generator on the *visualizer* subfolder to generates Visual Studio project files on Windows, and a Makefile on Linux / OS X.

The visualization tool was tested on
- Windows (Visual Studio 2013-2017 Win64)
- Linux (GCC 6.3.1)

## License

The new code introduced by this project is licensed under the GNU General Public License (Version 3). Please consult the bundled LICENSE file for the full license text.
