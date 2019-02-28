## Practical Path Guiding for Efficient Light-Transport Simulation

This repository contains the authors' implementation of the guided unidirectional path tracer of the research paper ["Practical Path Guiding for Efficient Light-Transport Simulation" [Müller et al. 2017]](https://tom94.net). It also includes a visualization tool for the SD-Trees learned by the guided path tracer. The guided path tracer has been implemented in the [Mitsuba Physically Based Renderer](http://mitsuba-renderer.org) and the visualization tool with the [nanogui](https://github.com/wjakob/nanogui) library.

## Extensions

This repository contains extensions beyond what was presented in [Müller et al. 2017].
The extensions are
- filtered SD-tree splatting for increased robustness,
- support for next event estimation (NEE) in addition to path guiding, and
- automatic learning of the BSDF / SD-tree sampling ratio via gradient descent based on the theory of [Neural Importance Sampling [Müller et al. 2018]](https://tom94.net).

Since the above extensions significantly improve the algorithm, they are *enabled* by default.
Please consult the bundled KITCHEN-OLD scene or the Mitsuba GUI to see the parameters that need to be disabled for reproducing the results of Müller et al. [2017].

### No Support for Participating Media

The guided path tracer in this repository was not designed to handle participating media, although it could potentially be extended with little effort. In its current state, scenes containing participating media might converge slowly or not to the correct result at all.

## Scenes

The KITCHEN scene from the paper is included in this repository. It was originally modeled by [Jay-Artist on Blendswap](http://www.blendswap.com/user/Jay-Artist), converted into a Mitsuba scene by [Benedikt Bitterli](https://benedikt-bitterli.me/resources/), and then slightly modified by us. The scene is covered by the [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/).

The TORUS scene is available for download on the [Mitsuba website](http://mitsuba-renderer.org/download.html). It was created by Olesya Jakob.

The POOL scene—created by Ondřej Karlík—is bundled with the [public source code of the method by Vorba et al. [2014]](http://cgg.mff.cuni.cz/~jirka/papers/2014/olpm/index.htm).

## Implementation

- The guided path tracer is implemented as the `GuidedPathTracer` Mitsuba integrator.
- The visualization tool is implemented as a standalone program built on nanogui.

### Modifications to Mitsuba

- `BlockedRenderProcess` (*renderproc.cpp, renderproc.h*)
  - Allowed retrieving the total amount of blocks.
  - Disabled automatic progress bookkeeping.
- `GuidedPathTracer` (*guided_path.cpp*)
  - Added the guided path tracer implementing [Müller et al. 2017].
  - Additionally, implemented the following extensions that are not implemented in the paper:
    - Filtered SD-tree splatting.
    - Support for guiding with next event estimation (NEE) enabled.
    - Automatic learning of the BSDF / SD-tree sampling ratio via gradient descent based on the theory of [Neural Importance Sampling [Müller et al. 2018]](https://tom94.net).
- `ImageBlock` (*imageblock.h*)
  - Allowed querying the reconstruction filter.
- `MainWindow` (*mainwindow.cpp*)
  - Removed warning about orphaned rectangular work units (occured when multiple threads write into spatially overlapping blocks at the same time).
- General
  - Added `GuidedPathTracer` to *src/integrator/SConscript* (for compilation) and *src/mtsgui/resources/docs.xml* (for mtsgui).
  - Changed the Visual Studio 2010 project to a Visual Studio 2013 project to make our integrator compile.
  - Removed the Irawan BSDF to make mitsuba compile under newer GCC versions.
  - Fixed various issues of the PLY parser to make mitsuba compile under newer GCC versions and clang.

### Modifications to nanogui

- `ImageView` (*imageview.cpp, imageview.h*)
  - Changed the shader to display a false-color visualization of a given high-dynamic-range image.
- General
  - Removed `noexcept` qualifiers to make nanogui compile under Visual Studio 2013.
  - Removed `constexpr` qualifiers to make nanogui compile under Visual Studio 2013.

## Compilation

### Mitsuba

To compile the Mitsuba code, please follow the instructions from the [Mitsuba documentation](http://mitsuba-renderer.org/docs.html) (sections 4.1.1 through 4.6). Since our new code uses C++11 features, a slightly more recent compiler and dependencies than reported in the mitsuba documentation may be required. We only support compiling mitsuba with the [scons](https://www.scons.org) build system.

We tested our Mitsuba code on
- Windows (Visual Studio 2013 Win64, custom dependencies via `git clone https://github.com/Tom94/mitsuba-dependencies-windows mitsuba/dependencies`)
- macOS (High Sierra, custom dependencies via `git clone https://github.com/Tom94/mitsuba-dependencies-macOS mitsuba/dependencies`)
- Linux (GCC 6.3.1)

### Visualization Tool

The visualization tool, found in the *visualizer* subfolder, uses the [CMake](https://cmake.org/) build system. Simply invoke the CMake generator on the *visualizer* subfolder to generate Visual Studio project files on Windows, and a Makefile on Linux / OS X.

The visualization tool was tested on
- Windows (Visual Studio 2013-2017 Win64)
- macOS (High Sierra)
- Linux (GCC 6.3.1)

## License

The new code introduced by this project is licensed under the GNU General Public License (Version 3). Please consult the bundled LICENSE file for the full license text.

The bundled KITCHEN scene is governed by the [CC-BY 3.0 license](https://creativecommons.org/licenses/by/3.0/).
