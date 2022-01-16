mkdir -p Materials.app/Contents/MacOS

clang++ -o Materials.app/Contents/MacOS/Materials Materials.mm -framework Cocoa -framework QuartzCore -framework OpenGL
