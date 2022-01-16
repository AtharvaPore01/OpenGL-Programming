mkdir -p PerVertexLightingOnSphere.app/Contents/MacOS

clang++ -o PerVertexLightingOnSphere.app/Contents/MacOS/PerVertexLightingOnSphere PerVertexLightingOnSphere.mm -framework Cocoa -framework QuartzCore -framework OpenGL
