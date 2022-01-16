mkdir -p PerFragmentLightingOnSphere.app/Contents/MacOS

clang++ -o PerFragmentLightingOnSphere.app/Contents/MacOS/PerFragmentLightingOnSphere PerFragmentLightingOnSphere.mm -framework Cocoa -framework QuartzCore -framework OpenGL
