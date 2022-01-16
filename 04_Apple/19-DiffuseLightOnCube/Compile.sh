mkdir -p DiffuselightOnCube.app/Contents/MacOS

clang++ -o DiffuselightOnCube.app/Contents/MacOS/DiffuselightOnCube DiffuselightOnCube.mm -framework Cocoa -framework QuartzCore -framework OpenGL
