mkdir -p Meshes.app/Contents/MacOS

clang++ -o Meshes.app/Contents/MacOS/Meshes Meshes.mm -framework Cocoa -framework QuartzCore -framework OpenGL
