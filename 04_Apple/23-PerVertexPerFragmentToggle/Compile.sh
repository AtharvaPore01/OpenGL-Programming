mkdir -p PerVertexPerFragmentToggle.app/Contents/MacOS

clang++ -o PerVertexPerFragmentToggle.app/Contents/MacOS/PerVertexPerFragmentToggle PerVertexPerFragmentToggle.mm -framework Cocoa -framework QuartzCore -framework OpenGL
