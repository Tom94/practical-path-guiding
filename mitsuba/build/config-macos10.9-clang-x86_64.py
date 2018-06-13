BUILDDIR       = '#build/release'
DISTDIR        = '#Mitsuba.app'
CXX            = 'clang++'
CC             = 'clang'
CCFLAGS        = ['-arch', 'x86_64', '-mmacosx-version-min=10.9', '-march=nocona', '-msse2', '-ftree-vectorize', '-funsafe-math-optimizations', '-fno-math-errno', '-isysroot', 'dependencies/MacOSX10.9.sdk', '-O3', '-Wall', '-g', '-pipe', '-DMTS_DEBUG', '-DSINGLE_PRECISION', '-DSPECTRUM_SAMPLES=3', '-DMTS_SSE', '-DMTS_HAS_COHERENT_RT', '-fstrict-aliasing', '-fsched-interblock', '-fvisibility=hidden', '-ftemplate-depth=512', '-stdlib=libc++', '-Wno-unused-local-typedef']
CXXFLAGS       = ['-arch', 'x86_64', '-mmacosx-version-min=10.9', '-march=nocona', '-msse2', '-ftree-vectorize', '-funsafe-math-optimizations', '-fno-math-errno', '-isysroot', 'dependencies/MacOSX10.9.sdk', '-O3', '-Wall', '-g', '-pipe', '-DMTS_DEBUG', '-DSINGLE_PRECISION', '-DSPECTRUM_SAMPLES=3', '-DMTS_SSE', '-DMTS_HAS_COHERENT_RT', '-fstrict-aliasing', '-fsched-interblock', '-fvisibility=hidden', '-ftemplate-depth=512', '-stdlib=libc++', '-Wno-unused-local-typedef', '-Wno-deprecated-register', '-std=c++11']
LINKFLAGS      = ['-framework', 'OpenGL', '-framework', 'Cocoa', '-arch', 'x86_64', '-mmacosx-version-min=10.9', '-Wl,-syslibroot,dependencies/MacOSX10.9.sdk', '-Wl,-headerpad,128', '-stdlib=libc++', '-std=c++11']
BASEINCLUDE    = ['#include', '#dependencies/include']
BASELIBDIR     = ['#dependencies/lib']
BASELIB        = ['m', 'pthread', 'Half']
OEXRINCLUDE    = ['#dependencies/include/OpenEXR']
OEXRLIB        = ['IlmImf', 'Imath', 'Iex', 'z']
PNGINCLUDE     = ['#dependencies/include/libpng']
PNGLIB         = ['png16']
JPEGINCLUDE    = ['#dependencies/include/libjpeg']
JPEGLIB        = ['jpeg']
XERCESLIB      = ['xerces-c']
GLLIB          = ['GLEWmx', 'objc']
GLFLAGS        = ['-DGLEW_MX']
BOOSTINCLUDE   = ['#dependencies']
BOOSTLIB       = ['boost_filesystem', 'boost_system', 'boost_thread']
COLLADAINCLUDE = ['#dependencies/include/collada-dom', '#dependencies/include/collada-dom/1.4']
COLLADALIB     = ['collada14dom24']
QTDIR          = '#dependencies'
FFTWLIB        = ['fftw3']
