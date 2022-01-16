mkdir -p DiffuseLightOnSphere.app/Contents/MacOS

clang++ -o DiffuseLightOnSphere.app/Contents/MacOS/DiffuseLightOnSphere DiffuseLightOnSphere.mm -framework Cocoa -framework QuartzCore -framework OpenGL
